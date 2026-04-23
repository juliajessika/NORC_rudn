/**
 * QuPath Groovy script
 *
 * Export CellSighter-style data from current annotations,
 * run CellSighter evaluation with an already-trained model via conda,
 * then import predictions back into QuPath cell objects.
 *
 * Expected CellSighter structure:
 *   {root}/CellTypes/data/images/<image_id>.tiff
 *   {root}/CellTypes/cells/<image_id>.tiff
 *   {root}/CellTypes/cells2labels/<image_id>.txt
 *   {root}/channels.txt
 *
 * IMPORTANT:
 *   cell_id is handled by explicit mapping:
 *     imageIdToCellMap[image_id][cell_id] = PathObject
 *   instead of relying on list index (cell_id - 1).
 */

import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RoiTools
import qupath.lib.common.GeneralTools
import qupath.lib.objects.PathObject

import java.awt.Color
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.Shape
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.awt.image.WritableRaster
import javax.imageio.ImageIO

// ============================================================
// USER SETTINGS
// ============================================================

double downsample = 1.0
String exportRootName = "cellsighter_export_inference1"
String rawExt = ".tiff"
String maskExt = ".tiff"
boolean exportSelectedAnnotationsOnly = false
boolean skipUnclassifiedCells = false
boolean simplifyClassNames = true
double minCellAreaPx = 0.0
boolean useRecursiveChildren = true
boolean prependBackgroundLabelInTxt = true
boolean skipEmptyAnnotations = true
String imageIdPrefix = "FOV"
boolean maskRawOutsideAnnotation = false
boolean exportNativeRaw = true

int cropInputSize = 60
int cropSize = 128
int epochMax = 50
double learningRate = 0.001
boolean toPad = true
boolean sampleBatch = true
boolean aug = false
Integer sizeData = null
int batchSize = 32
int numWorkers = 0
def blacklist = []
boolean writeHierarchyMatch = true
boolean writeHierarchyMatchFromClasses = true

boolean runCellSighterEvaluation = true

String cellSighterEvalPy = "C:/Users/julia/CellSighter/eval.py"
String condaExe = "C:/Users/julia/anaconda3/Scripts/conda.exe"
String condaEnvName = "cellsighter"
String trainedModelPath = "C:/Users/julia/OneDrive/Desktop/RUDN/TileServer/SampleProject/cellsighter_export/weights_49_count.pth"
String trainedClassMapCsv = "C:/Users/julia/OneDrive/Desktop/RUDN/TileServer/SampleProject/cellsighter_export/class_map.csv"

boolean writeAllMinusOneLabelsForEval = true
boolean writePredictedPathClass = true
boolean writePredictionMeasurements = true
boolean prefixImportedClassNames = false

// ============================================================
// HELPERS
// ============================================================

def cleanClassName = { pathClassObj ->
    if (pathClassObj == null)
        return null
    String s = pathClassObj.toString()
    if (!simplifyClassNames)
        return s

    def m = (s =~ /^.*\((.*)\)$/)
    if (m.matches())
        return m[0][1].trim()
    return s
}

def safeName = { String s ->
    if (s == null || s.isBlank())
        return "unnamed"
    return s.replaceAll("[^A-Za-z0-9._-]", "_")
}

def csvEscape = { String s ->
    if (s == null)
        return ""
    String t = s.replace("\"", "\"\"")
    if (t.contains(",") || t.contains("\"") || t.contains("\n") || t.contains("\r"))
        return "\"${t}\""
    return t
}

def collectDescendants
collectDescendants = { obj ->
    def out = []
    def kids = obj.getChildObjects()
    if (kids == null || kids.isEmpty())
        return out
    kids.each { k ->
        out << k
        out.addAll(collectDescendants(k))
    }
    return out
}

def isUsableCellLikeObject = { obj ->
    if (obj == null)
        return false
    def roi = obj.getROI()
    if (roi == null || !roi.isArea())
        return false
    if (minCellAreaPx > 0 && roi.getArea() < minCellAreaPx)
        return false
    if (!writeAllMinusOneLabelsForEval && skipUnclassifiedCells && obj.getPathClass() == null)
        return false
    return true
}

def getCellsForParent = { parentAnn ->
    def childObjects = useRecursiveChildren ? collectDescendants(parentAnn) : parentAnn.getChildObjects()
    if (childObjects == null || childObjects.isEmpty())
        return []

    def cells = childObjects.findAll { obj -> isUsableCellLikeObject(obj) }

    cells = cells.sort { a, b ->
        def ra = a.getROI()
        def rb = b.getROI()
        double ay = ra.getCentroidY()
        double by = rb.getCentroidY()
        if (ay != by)
            return ay <=> by
        double ax = ra.getCentroidX()
        double bx = rb.getCentroidX()
        return ax <=> bx
    }

    return cells
}

def writeTextFile = { String path, String text ->
    def f = new File(path)
    f.parentFile?.mkdirs()
    f.text = text
}

def appendTextFile = { String path, String text ->
    def f = new File(path)
    f.parentFile?.mkdirs()
    f << text
}

def normalizeForJsonPath = { String p ->
    if (p == null)
        return ""
    return p.replace("\\", "/")
}

def jsonEscape = { String s ->
    if (s == null)
        return ""
    return s
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\b", "\\b")
            .replace("\f", "\\f")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
}

def jsonString = { String s ->
    return "\"" + jsonEscape(s == null ? "" : s) + "\""
}

def jsonArrayOfStrings = { list ->
    return "[" + list.collect { jsonString(it == null ? "" : it.toString()) }.join(", ") + "]"
}

def makeStandardRGBImage = { BufferedImage src ->
    BufferedImage rgb = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB)
    Graphics2D g2d = rgb.createGraphics()
    g2d.setColor(Color.BLACK)
    g2d.fillRect(0, 0, rgb.getWidth(), rgb.getHeight())
    g2d.drawImage(src, 0, 0, null)
    g2d.dispose()
    return rgb
}

def applyAnnotationMaskToRenderedCrop = { BufferedImage src, parentROI, int x0, int y0, int w, int h ->
    int outW = src.getWidth()
    int outH = src.getHeight()

    BufferedImage out = new BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB)
    Graphics2D g2d = out.createGraphics()
    g2d.setColor(Color.BLACK)
    g2d.fillRect(0, 0, outW, outH)

    double scaleX = outW / (double)w
    double scaleY = outH / (double)h

    AffineTransform tx = new AffineTransform()
    tx.scale(scaleX, scaleY)
    tx.translate(-x0, -y0)

    Shape shape = RoiTools.getShape(parentROI)
    Shape transformedShape = tx.createTransformedShape(shape)

    g2d.setClip(transformedShape)
    g2d.drawImage(src, 0, 0, null)
    g2d.dispose()
    return out
}

def deleteIfExists = { String path ->
    def f = new File(path)
    if (f.exists())
        f.delete()
}

def cleanupPreviousSampleFiles = { String imagesDir, String cellsDir, String cells2labelsDir, String metaDir, String imageId ->
    [
        buildFilePath(imagesDir, imageId + ".tif"),
        buildFilePath(imagesDir, imageId + ".tiff"),
        buildFilePath(cellsDir, imageId + ".tif"),
        buildFilePath(cellsDir, imageId + ".tiff"),
        buildFilePath(cells2labelsDir, imageId + ".txt"),
        buildFilePath(metaDir, imageId + "_cell_table.csv")
    ].each { p ->
        deleteIfExists(p)
    }
}

def getChannelNames = { server ->
    def out = []
    try {
        def metadata = server.getMetadata()
        def channels = metadata?.getChannels()
        if (channels != null && !channels.isEmpty()) {
            channels.eachWithIndex { ch, i ->
                String nm = ch?.getName()
                if (nm == null || nm.isBlank())
                    nm = "Channel_${i + 1}"
                out << nm
            }
        }
    } catch (Exception e) {
    }

    if (out.isEmpty()) {
        try {
            int n = server.nChannels()
            for (int i = 0; i < n; i++)
                out << "Channel_${i + 1}"
        } catch (Exception e) {
            out << "Channel_1"
        }
    }
    return out
}

def parseSimpleCsvLine = { String line ->
    def result = []
    if (line == null)
        return result

    StringBuilder sb = new StringBuilder()
    boolean inQuotes = false
    for (int i = 0; i < line.length(); i++) {
        char c = line.charAt(i)
        if (c == '"') {
            if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                sb.append('"')
                i++
            } else {
                inQuotes = !inQuotes
            }
        } else if (c == ',' && !inQuotes) {
            result << sb.toString()
            sb.setLength(0)
        } else {
            sb.append(c)
        }
    }
    result << sb.toString()
    return result
}

def readClassMapCsv = { String path ->
    def idToClass = [:]
    def classToId = [:]

    def f = new File(path)
    if (!f.exists())
        throw new RuntimeException("Training class_map.csv not found: " + path)

    def lines = f.readLines()
    if (lines.isEmpty())
        throw new RuntimeException("Training class_map.csv is empty: " + path)

    lines.drop(1).each { line ->
        if (line == null || line.trim().isEmpty())
            return
        def cols = parseSimpleCsvLine(line)
        if (cols.size() < 2)
            return
        String className = cols[0]
        int classId = cols[1].trim().toInteger()
        idToClass[classId] = className
        classToId[className] = classId
    }

    return [idToClass: idToClass, classToId: classToId]
}

def runCommandAndCapture = { List<String> cmd, File workDir = null ->
    println "Running command: " + cmd.join(" ")

    def pb = new ProcessBuilder(cmd)
    if (workDir != null)
        pb.directory(workDir)
    pb.redirectErrorStream(true)

    def proc = pb.start()
    def lines = []
    proc.inputStream.withReader("UTF-8") { r ->
        String line
        while ((line = r.readLine()) != null) {
            println line
            lines << line
        }
    }
    int exitCode = proc.waitFor()
    return [exitCode: exitCode, output: lines.join("\n")]
}

def findValResultsFile = { String runDir ->
    def dir = new File(runDir)
    if (!dir.exists())
        return null

    def candidates = dir.listFiles()?.findAll {
        it.isFile() && (it.name == "val_results.csv" || it.name.startsWith("val_results"))
    } ?: []

    if (candidates.isEmpty())
        return null

    candidates = candidates.sort { a, b -> b.lastModified() <=> a.lastModified() }
    return candidates[0]
}

// Parse string like "[0.1, 0.2, 0.7]" into List<Double>
def parseProbList = { String s ->
    def out = []
    if (s == null)
        return out
    String t = s.trim()
    if (t.isEmpty())
        return out
    if (t.startsWith("["))
        t = t.substring(1)
    if (t.endsWith("]"))
        t = t.substring(0, t.length() - 1)
    if (t.trim().isEmpty())
        return out

    t.split(",").each { part ->
        String p = part.trim()
        if (!p.isEmpty()) {
            try {
                out << Double.parseDouble(p)
            } catch (Exception e) {
            }
        }
    }
    return out
}

// ============================================================
// INITIAL SETUP
// ============================================================

def imageData = getCurrentImageData()
def server = imageData.getServer()
def hierarchy = imageData.getHierarchy()
def imageName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

if (getProject() == null) {
    print "ERROR: Please open a QuPath project first."
    return
}

def channelNames = getChannelNames(server)

def mapInfo = readClassMapCsv(trainedClassMapCsv)
def trainedIdToClass = mapInfo.idToClass
def trainedClassToId = mapInfo.classToId

print "Loaded training class map from: ${trainedClassMapCsv}"
trainedIdToClass.keySet().sort().each { id ->
    print "  ${id} -> ${trainedIdToClass[id]}"
}

def exportRoot = buildFilePath(PROJECT_BASE_DIR, exportRootName)
def cellTypesRoot = buildFilePath(exportRoot, "CellTypes")
def imagesDir = buildFilePath(cellTypesRoot, "data", "images")
def cellsDir = buildFilePath(cellTypesRoot, "cells")
def cells2labelsDir = buildFilePath(cellTypesRoot, "cells2labels")
def metaDir = buildFilePath(exportRoot, "metadata")

mkdirs(exportRoot)
mkdirs(cellTypesRoot)
mkdirs(imagesDir)
mkdirs(cellsDir)
mkdirs(cells2labelsDir)
mkdirs(metaDir)

// ============================================================
// GET PARENT ANNOTATIONS
// ============================================================

def parentAnnotations
if (exportSelectedAnnotationsOnly) {
    parentAnnotations = getSelectedObjects().findAll { it.isAnnotation() && it.getROI() != null && it.getROI().isArea() }
} else {
    parentAnnotations = getAnnotationObjects().findAll { it.getROI() != null && it.getROI().isArea() }
}

if (parentAnnotations.isEmpty()) {
    print "ERROR: No parent annotations found."
    return
}

print "Found ${parentAnnotations.size()} parent annotation(s) to export."

def imageIdToCellMap = [:]

def channelsPath = buildFilePath(exportRoot, "channels.txt")
writeTextFile(channelsPath, channelNames.join("\n") + "\n")

def classMapPath = buildFilePath(exportRoot, "class_map.csv")
writeTextFile(classMapPath, "class_name,class_id\n")
trainedIdToClass.keySet().sort().each { id ->
    appendTextFile(classMapPath, csvEscape(trainedIdToClass[id]) + "," + id + "\n")
}

def exportSummaryPath = buildFilePath(exportRoot, "export_summary.csv")
writeTextFile(exportSummaryPath,
        "image_id,source_image,parent_annotation,annotation_index,cell_count,raw_path,mask_path,labels_path\n"
)

def exportedImageIds = []
int exportedCount = 0

// ============================================================
// EXPORT EACH ANNOTATION
// ============================================================

parentAnnotations.eachWithIndex { parentAnn, annIdx ->

    def parentROI = parentAnn.getROI()
    String parentLabel = parentAnn.getName()
    if (parentLabel == null || parentLabel.isBlank())
        parentLabel = "annotation_${annIdx + 1}"

    def cells = getCellsForParent(parentAnn)
    if (cells == null)
        cells = []

    if (cells.isEmpty()) {
        String msg = "WARNING: annotation '${parentLabel}' has no suitable child cell objects."
        if (skipEmptyAnnotations) {
            print msg + " Skipping."
            return
        } else {
            print msg + " Exporting empty label/mask files."
        }
    }

    String imageId = safeName("${imageIdPrefix}_${imageName}_${parentLabel}")
    imageIdToCellMap[imageId] = [:]

    cleanupPreviousSampleFiles(imagesDir, cellsDir, cells2labelsDir, metaDir, imageId)

    print ""
    print "====================================="
    print "Exporting annotation ${annIdx + 1}/${parentAnnotations.size()}: ${parentLabel}"
    print "Image ID: ${imageId}"
    print "Cells kept: ${cells.size()}"

    int x0 = (int)Math.floor(parentROI.getBoundsX())
    int y0 = (int)Math.floor(parentROI.getBoundsY())
    int w  = (int)Math.ceil(parentROI.getBoundsWidth())
    int h  = (int)Math.ceil(parentROI.getBoundsHeight())

    if (w <= 0 || h <= 0) {
        print "WARNING: annotation '${parentLabel}' produced non-positive source size. Skipping."
        return
    }

    print "Region: x=${x0}, y=${y0}, w=${w}, h=${h}, downsample=${downsample}"

    def rawPath = buildFilePath(imagesDir, imageId + rawExt)
    def instanceMaskPath = buildFilePath(cellsDir, imageId + maskExt)
    def cells2labelsPath = buildFilePath(cells2labelsDir, imageId + ".txt")
    def cellTablePath = buildFilePath(metaDir, imageId + "_cell_table.csv")

    def request = RegionRequest.createInstance(server.getPath(), downsample, x0, y0, w, h)

    int outW
    int outH

    BufferedImage previewBuffered
    try {
        previewBuffered = server.readBufferedImage(request)
    } catch (Exception e) {
        print "ERROR: failed to read preview image for annotation '${parentLabel}': ${e.getMessage()}"
        return
    }

    if (previewBuffered == null) {
        print "ERROR: server.readBufferedImage(...) returned null for annotation '${parentLabel}'."
        return
    }

    outW = previewBuffered.getWidth()
    outH = previewBuffered.getHeight()

    if (outW <= 0 || outH <= 0) {
        print "ERROR: exported raw image has invalid size ${outW} x ${outH}."
        return
    }

    try {
        if (exportNativeRaw) {
            writeImageRegion(server, request, rawPath)
        } else {
            BufferedImage rawToWrite = makeStandardRGBImage(previewBuffered)
            if (maskRawOutsideAnnotation) {
                rawToWrite = applyAnnotationMaskToRenderedCrop(rawToWrite, parentROI, x0, y0, w, h)
            }
            boolean rawOk = ImageIO.write(rawToWrite, "TIFF", new File(rawPath))
            if (!rawOk || !new File(rawPath).exists()) {
                print "ERROR: failed to write raw TIFF: ${rawPath}"
                return
            }
        }
    } catch (Exception e) {
        print "ERROR: failed to export raw image for annotation '${parentLabel}': ${e.getMessage()}"
        return
    }

    if (!new File(rawPath).exists()) {
        print "ERROR: raw file was not created: ${rawPath}"
        return
    }

    print "Actual exported raw size used for mask: ${outW} x ${outH}"

    BufferedImage instanceMask = new BufferedImage(outW, outH, BufferedImage.TYPE_USHORT_GRAY)
    WritableRaster instanceRaster = instanceMask.getRaster()

    double scaleX = outW / (double)w
    double scaleY = outH / (double)h

    AffineTransform tx = new AffineTransform()
    tx.scale(scaleX, scaleY)
    tx.translate(-x0, -y0)

    BufferedImage tmp = new BufferedImage(outW, outH, BufferedImage.TYPE_BYTE_GRAY)

    if (prependBackgroundLabelInTxt) {
        writeTextFile(cells2labelsPath, "-1\n")
    } else {
        writeTextFile(cells2labelsPath, "")
    }

    writeTextFile(cellTablePath,
        "cell_id,class_id,class_name,centroid_x_global,centroid_y_global,centroid_x_local,centroid_y_local,object_name\n"
    )

    int runningCellId = 1

    cells.eachWithIndex { cellObj, idx ->

        def roi = cellObj.getROI()
        if (roi == null)
            return

        int classId
        String className

        if (writeAllMinusOneLabelsForEval) {
            classId = -1
            className = ""
        } else {
            className = cleanClassName(cellObj.getPathClass())
            classId = (className != null && trainedClassToId.containsKey(className)) ? (int)trainedClassToId[className] : -1
        }

        Shape shape = RoiTools.getShape(roi)
        Shape transformedShape = tx.createTransformedShape(shape)

        def gTmp = tmp.createGraphics()
        gTmp.setBackground(Color.BLACK)
        gTmp.clearRect(0, 0, outW, outH)
        gTmp.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        gTmp.setColor(Color.WHITE)
        gTmp.fill(transformedShape)
        gTmp.dispose()

        def tmpRaster = tmp.getRaster()
        def bounds = transformedShape.getBounds()

        int bx0 = Math.max(0, (int)Math.floor(bounds.x))
        int by0 = Math.max(0, (int)Math.floor(bounds.y))
        int bx1 = Math.min(outW, (int)Math.ceil(bounds.x + bounds.width))
        int by1 = Math.min(outH, (int)Math.ceil(bounds.y + bounds.height))

        if (bx1 > bx0 && by1 > by0) {
            for (int yy = by0; yy < by1; yy++) {
                for (int xx = bx0; xx < bx1; xx++) {
                    int v = tmpRaster.getSample(xx, yy, 0)
                    if (v > 0) {
                        instanceRaster.setSample(xx, yy, 0, runningCellId)
                    }
                }
            }
        }

        appendTextFile(cells2labelsPath, "${classId}\n")

        String objName = cellObj.getName()
        if (objName == null)
            objName = ""

        double cxGlobal = roi.getCentroidX()
        double cyGlobal = roi.getCentroidY()
        double cxLocal = (cxGlobal - x0) * scaleX
        double cyLocal = (cyGlobal - y0) * scaleY

        appendTextFile(cellTablePath, [
            runningCellId,
            classId,
            csvEscape(className == null ? "" : className),
            cxGlobal,
            cyGlobal,
            cxLocal,
            cyLocal,
            csvEscape(objName)
        ].join(",") + "\n")

        imageIdToCellMap[imageId][runningCellId] = cellObj

        runningCellId++

        if ((idx + 1) % 100 == 0) {
            print "Rasterized ${idx + 1}/${cells.size()} cells"
        }
    }

    boolean maskOk = ImageIO.write(instanceMask, "TIFF", new File(instanceMaskPath))
    if (!maskOk || !new File(instanceMaskPath).exists()) {
        print "ERROR: failed to write instance mask TIFF: ${instanceMaskPath}"
        return
    }

    appendTextFile(exportSummaryPath, [
        csvEscape(imageId),
        csvEscape(imageName),
        csvEscape(parentLabel),
        (annIdx + 1),
        cells.size(),
        csvEscape(rawPath),
        csvEscape(instanceMaskPath),
        csvEscape(cells2labelsPath)
    ].join(",") + "\n")

    exportedImageIds << imageId

    print "Saved:"
    print "  raw image     -> ${rawPath}"
    print "  instance mask -> ${instanceMaskPath}"
    print "  labels txt    -> ${cells2labelsPath}"
    print "  cell table    -> ${cellTablePath}"

    exportedCount++
}

// ============================================================
// WRITE EVAL CONFIG.JSON
// ============================================================

if (exportedImageIds.isEmpty()) {
    print "ERROR: No image IDs were exported, so config.json was not created."
    return
}

def hierarchyMatch = [:]
if (writeHierarchyMatch) {
    trainedClassToId.each { cls, id ->
        hierarchyMatch["${id}"] = writeHierarchyMatchFromClasses ? cls : "${id}"
    }
}

def configPath = buildFilePath(exportRoot, "config.json")

def trainSet = []
def valSet = new ArrayList(exportedImageIds)

def jsonLines = []
jsonLines << "{"
jsonLines << "  \"crop_input_size\": ${cropInputSize},"
jsonLines << "  \"crop_size\": ${cropSize},"
jsonLines << "  \"root_dir\": " + jsonString(normalizeForJsonPath(exportRoot)) + ","
jsonLines << "  \"train_set\": " + jsonArrayOfStrings(trainSet) + ","
jsonLines << "  \"val_set\": " + jsonArrayOfStrings(valSet) + ","
jsonLines << "  \"num_classes\": ${trainedClassToId.size()},"
jsonLines << "  \"epoch_max\": ${epochMax},"
jsonLines << "  \"lr\": ${learningRate},"
jsonLines << "  \"to_pad\": ${toPad},"
jsonLines << "  \"blacklist\": " + jsonArrayOfStrings(blacklist) + ","
jsonLines << "  \"channels_path\": " + jsonString(normalizeForJsonPath(channelsPath)) + ","
jsonLines << "  \"weight_to_eval\": " + jsonString(normalizeForJsonPath(trainedModelPath)) + ","
jsonLines << "  \"sample_batch\": ${sampleBatch},"

jsonLines << "  \"hierarchy_match\": {"
def hmEntries = hierarchyMatch.collect { k, v ->
    "    " + jsonString(k.toString()) + ": " + jsonString(v.toString())
}
if (!hmEntries.isEmpty()) {
    for (int i = 0; i < hmEntries.size(); i++) {
        jsonLines << hmEntries[i] + (i < hmEntries.size() - 1 ? "," : "")
    }
}
jsonLines << "  },"

jsonLines << "  \"batch_size\": ${batchSize},"
jsonLines << "  \"num_workers\": ${numWorkers},"
jsonLines << "  \"aug\": ${aug}" + (sizeData != null ? "," : "")

if (sizeData != null) {
    jsonLines << "  \"size_data\": ${sizeData}"
}

jsonLines << "}"

writeTextFile(configPath, jsonLines.join("\n"))

print ""
print "Config written: ${configPath}"

// ============================================================
// RUN CELLSIGHTER EVAL THROUGH CONDA
// ============================================================

File resultsCsv = null

if (runCellSighterEvaluation) {
    def cmd = [
        condaExe,
        "run",
        "-n", condaEnvName,
        "python",
        cellSighterEvalPy,
        "--base_path=" + new File(exportRoot).getAbsolutePath()
    ]

    def runInfo = runCommandAndCapture(cmd, new File(exportRoot))
    if (runInfo.exitCode != 0) {
        print "ERROR: CellSighter eval failed with exit code ${runInfo.exitCode}"
        return
    }

    resultsCsv = findValResultsFile(exportRoot)
    if (resultsCsv == null || !resultsCsv.exists()) {
        print "ERROR: Could not find val_results output inside: ${exportRoot}"
        return
    }

    print "Found CellSighter results: ${resultsCsv.getAbsolutePath()}"
} else {
    resultsCsv = new File(buildFilePath(exportRoot, "val_results.csv"))
    if (!resultsCsv.exists()) {
        print "ERROR: runCellSighterEvaluation=false, but val_results.csv not found at ${resultsCsv.getAbsolutePath()}"
        return
    }
}

// ============================================================
// IMPORT PREDICTIONS BACK INTO QUPATH
// ============================================================

def resultLines = resultsCsv.readLines()
if (resultLines.size() < 2) {
    print "ERROR: Results file is empty or has no data rows: ${resultsCsv.getAbsolutePath()}"
    return
}

def header = parseSimpleCsvLine(resultLines[0])
def colIndex = [:]
header.eachWithIndex { c, i -> colIndex[c] = i }

def requiredCols = ["pred", "pred_prob", "cell_id", "image_id"]
def missingRequiredCols = requiredCols.findAll { !colIndex.containsKey(it) }
if (!missingRequiredCols.isEmpty()) {
    print "ERROR: Results CSV missing required columns: ${missingRequiredCols}"
    print "Header: ${header}"
    return
}

int importedCount = 0
int missingImageCount = 0
int missingCellCount = 0
int parseErrorCount = 0
int unknownPredClassCount = 0

resultLines.drop(1).eachWithIndex { line, lineIdx ->
    if (line == null || line.trim().isEmpty())
        return

    def cols = parseSimpleCsvLine(line)
    if (cols.size() < header.size()) {
        print "WARNING: skipping malformed results row ${lineIdx + 2}"
        return
    }

    String imageId = cols[colIndex["image_id"]]
    String predStr = cols[colIndex["pred"]]
    String predProbStr = cols[colIndex["pred_prob"]]
    String cellIdStr = cols[colIndex["cell_id"]]
    String probListStr = colIndex.containsKey("prob_list") ? cols[colIndex["prob_list"]] : ""

    if (imageId == null || imageId.isEmpty()) {
        print "WARNING: result row ${lineIdx + 2} has empty image_id"
        return
    }

    def cellMapForImage = imageIdToCellMap[imageId]
    if (cellMapForImage == null) {
        missingImageCount++
        print "WARNING: no exported cell map for image_id='${imageId}'"
        return
    }

    int cellId
    int predId
    double predProb = Double.NaN

    try {
        cellId = cellIdStr.toInteger()
        predId = predStr.toInteger()
        predProb = predProbStr.toDouble()
    } catch (Exception e) {
        parseErrorCount++
        print "WARNING: could not parse numeric values in results row ${lineIdx + 2}"
        return
    }

    def cellObj = cellMapForImage[cellId]
    if (cellObj == null) {
        missingCellCount++
        print "WARNING: cell_id=${cellId} not found for image_id='${imageId}'"
        return
    }

    String predictedClassName = trainedIdToClass[predId]
    if (predictedClassName == null) {
        unknownPredClassCount++
        predictedClassName = "class_" + predId
    }

    String finalClassName = prefixImportedClassNames ? "CellSighter: " + predictedClassName : predictedClassName

    if (writePredictedPathClass) {
        def pathClass = getPathClass(finalClassName)
        cellObj.setPathClass(pathClass)
    }

    if (writePredictionMeasurements) {
        def ml = cellObj.getMeasurementList()
        ml.put("CellSighter pred_id", (double)predId)
        ml.put("CellSighter pred_prob", predProb)

        def probs = parseProbList(probListStr)
        probs.eachWithIndex { p, i ->
            ml.put("CellSighter prob class ${i}", (double)p)
        }

        ml.close()
    }

    cellObj.setName(predictedClassName)

    importedCount++
}

fireHierarchyUpdate()
hierarchy.fireHierarchyChangedEvent(this)

// ============================================================
// DONE
// ============================================================

print ""
print "====================================="
print "ALL DONE."
print "Export root: ${exportRoot}"
print "Images exported: ${exportedCount}"
print "Channels file: ${channelsPath}"
print "Config file: ${configPath}"
print "Results file: ${resultsCsv.getAbsolutePath()}"
print "Predictions imported into QuPath: ${importedCount}"
print "Missing image rows: ${missingImageCount}"
print "Missing cell rows: ${missingCellCount}"
print "Parse error rows: ${parseErrorCount}"
print "Unknown predicted class ids: ${unknownPredClassCount}"
print "Raw export mode: " + (exportNativeRaw ? "native channel-wise TIFF via writeImageRegion(...)" : "rendered RGB fallback")
