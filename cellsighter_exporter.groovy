/**
 * QuPath Groovy script
 *
 * Export CellSighter-style training data from parent annotations
 * that contain classified child cell objects.
 *
 * CellSighter expects:
 *   {root}/CellTypes/data/images/<image_id>.tif
 *   {root}/CellTypes/cells/<image_id>.tif
 *   {root}/CellTypes/cells2labels/<image_id>.txt
 *   {root}/channels.txt
 *
 * This script also writes:
 *   {root}/class_map.csv
 *   {root}/export_summary.csv
 *   {root}/train_set.txt
 *   {root}/val_set.txt
 *   {root}/config.json
 *   {root}/metadata/<image_id>_cell_table.csv
 *
 * Important:
 * - Class IDs are created GLOBALLY across the whole export.
 * - Instance mask uses cell IDs 1..N for each exported image.
 * - labels txt contains one row per cell in cell-id order.
 * - If prependBackgroundLabelInTxt = true, line 0 is -1, line 1 is cell 1, etc.
 *
 * Caveat:
 * - CellSighter expects raw HxWxC multiplex data.
 * - writeImageRegion(...) may export appropriately for your data,
 *   but you should verify exported TIFFs preserve the intended channels.
 */

import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RoiTools
import qupath.lib.common.GeneralTools

import java.awt.Color
import java.awt.RenderingHints
import java.awt.Shape
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.awt.image.WritableRaster
import javax.imageio.ImageIO
import java.util.Random
import java.util.Collections

// =====================================
// USER SETTINGS
// =====================================

// Export resolution
double downsample = 1.0

// Output folder inside project
String exportRootName = "cellsighter_export"

// Image extension for exported raw crops
String rawExt = ".tif"

// Instance mask extension
String maskExt = ".tif"

// If true: export only selected annotations
// If false: export all annotations
boolean exportSelectedAnnotationsOnly = false

// If true: only export child objects that have a classification
boolean skipUnclassifiedCells = true

// If true: sanitize class names by removing "Cell (" ... ")"
boolean simplifyClassNames = true

// Minimum area in px^2 for child cell object ROI to keep
double minCellAreaPx = 0.0

// If true, include descendants recursively instead of only direct children
boolean useRecursiveChildren = true

// If true, write labels txt with an initial background row (-1)
// so that line index == cell ID directly (line 1 = cell 1, etc.)
boolean prependBackgroundLabelInTxt = true

// If true, skip annotations that contain 0 suitable cells
boolean skipEmptyAnnotations = true

// Optional prefix for exported image IDs
String imageIdPrefix = "FOV"

// -----------------------------
// CellSighter config settings
// -----------------------------

// Fraction of exported image IDs to place into validation set
double valFraction = 0.2

// Deterministic split seed
long splitSeed = 12345L

// Training config defaults
int cropInputSize = 60
int cropSize = 128
int epochMax = 50
double learningRate = 0.001
boolean toPad = false
boolean sampleBatch = true
boolean aug = true

// Optional, set to null to omit from config.json
Integer sizeData = null

// Safer default on Windows
int batchSize = 32
int numWorkers = 0

// Channels to exclude from training if needed
def blacklist = []

// If true, hierarchy_match will map each class id to its own class name
// Example: "0":"T-lymph", "1":"Melanocyte"
boolean writeHierarchyMatchFromClasses = true

// If false, writes empty hierarchy_match {}
boolean writeHierarchyMatch = true

// =====================================
// HELPERS
// =====================================

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

def getCellsForParent = { parentAnn ->
    def childObjects = useRecursiveChildren ? collectDescendants(parentAnn) : parentAnn.getChildObjects()
    if (childObjects == null || childObjects.isEmpty())
        return []

    return childObjects.findAll { obj ->
        def roi = obj.getROI()
        if (roi == null || !roi.isArea())
            return false
        if (minCellAreaPx > 0 && roi.getArea() < minCellAreaPx)
            return false
        if (skipUnclassifiedCells && obj.getPathClass() == null)
            return false
        return true
    }
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

// =====================================
// INITIAL SETUP
// =====================================

def imageData = getCurrentImageData()
def server = imageData.getServer()
def imageName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

if (getProject() == null) {
    print "ERROR: Please open a QuPath project first."
    return
}

// Root structure required by CellSighter
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

// =====================================
// GET PARENT ANNOTATIONS
// =====================================

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

// =====================================
// FIRST PASS: COLLECT GLOBAL CLASS NAMES
// =====================================

def globalClassNames = new LinkedHashSet<String>()
def annotationToCells = [:]

parentAnnotations.eachWithIndex { parentAnn, annIdx ->
    def cells = getCellsForParent(parentAnn)
    annotationToCells[parentAnn] = cells

    cells.each { obj ->
        def cls = cleanClassName(obj.getPathClass())
        if (cls != null && !cls.isBlank())
            globalClassNames << cls
    }
}

if (globalClassNames.isEmpty()) {
    print "ERROR: No classified cells found in selected annotations."
    return
}

def sortedClassNames = globalClassNames.toList().sort()
def classToId = [:]
sortedClassNames.eachWithIndex { cls, idx ->
    classToId[cls] = idx
}

print ""
print "Global class mapping:"
classToId.each { k, v -> print "  ${k} -> ${v}" }

// Save global class map
def classMapPath = buildFilePath(exportRoot, "class_map.csv")
writeTextFile(classMapPath, "class_name,class_id\n")
classToId.each { cls, clsId ->
    appendTextFile(classMapPath, "${csvEscape(cls)},${clsId}\n")
}

// Try to write channels.txt from server metadata
def channelsPath = buildFilePath(exportRoot, "channels.txt")
def channelsText = new StringBuilder()

try {
    def metadata = server.getMetadata()
    def channels = metadata.getChannels()
    if (channels != null && !channels.isEmpty()) {
        channels.each { ch ->
            def nm = ch.getName()
            if (nm == null || nm.isBlank())
                nm = "Channel"
            channelsText.append(nm).append("\n")
        }
    } else {
        channelsText.append("Channel_1\n")
    }
} catch (Exception e) {
    channelsText.append("Channel_1\n")
}
writeTextFile(channelsPath, channelsText.toString())

// Global export summary
def exportSummaryPath = buildFilePath(exportRoot, "export_summary.csv")
writeTextFile(exportSummaryPath,
    "image_id,source_image,parent_annotation,annotation_index,cell_count,raw_path,mask_path,labels_path\n"
)

// Keep exported image IDs for config generation
def exportedImageIds = []

// =====================================
// SECOND PASS: EXPORT EACH ANNOTATION AS ONE IMAGE ID
// =====================================

int exportedCount = 0

parentAnnotations.eachWithIndex { parentAnn, annIdx ->

    def parentROI = parentAnn.getROI()
    String parentLabel = parentAnn.getName()
    if (parentLabel == null || parentLabel.isBlank())
        parentLabel = "annotation_${annIdx + 1}"

    def cells = annotationToCells[parentAnn]
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

    String imageId = safeName("${imageIdPrefix}_${imageName}_${annIdx + 1}_${parentLabel}")

    print ""
    print "====================================="
    print "Exporting annotation ${annIdx + 1}/${parentAnnotations.size()}: ${parentLabel}"
    print "Image ID: ${imageId}"
    print "Cells kept: ${cells.size()}"

    int x0 = (int)Math.floor(parentROI.getBoundsX())
    int y0 = (int)Math.floor(parentROI.getBoundsY())
    int w  = (int)Math.ceil(parentROI.getBoundsWidth())
    int h  = (int)Math.ceil(parentROI.getBoundsHeight())

    int outW = (int)Math.ceil(w / downsample)
    int outH = (int)Math.ceil(h / downsample)

    if (outW <= 0 || outH <= 0) {
        print "WARNING: annotation '${parentLabel}' produced non-positive output size. Skipping."
        return
    }

    print "Region: x=${x0}, y=${y0}, w=${w}, h=${h}, downsample=${downsample}"
    print "Output size: ${outW} x ${outH}"

    def rawPath = buildFilePath(imagesDir, imageId + rawExt)
    def instanceMaskPath = buildFilePath(cellsDir, imageId + maskExt)
    def cells2labelsPath = buildFilePath(cells2labelsDir, imageId + ".txt")
    def cellTablePath = buildFilePath(metaDir, imageId + "_cell_table.csv")

    def request = RegionRequest.createInstance(server.getPath(), downsample, x0, y0, w, h)
    writeImageRegion(server, request, rawPath)

    BufferedImage instanceMask = new BufferedImage(outW, outH, BufferedImage.TYPE_USHORT_GRAY)
    WritableRaster instanceRaster = instanceMask.getRaster()

    AffineTransform tx = new AffineTransform()
    tx.scale(1.0 / downsample, 1.0 / downsample)
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
        String className = cleanClassName(cellObj.getPathClass())
        int classId = (className != null && classToId.containsKey(className)) ? (int)classToId[className] : -1

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

        int bx0 = Math.max(0, (int)bounds.x)
        int by0 = Math.max(0, (int)bounds.y)
        int bx1 = Math.min(outW, (int)(bounds.x + bounds.width))
        int by1 = Math.min(outH, (int)(bounds.y + bounds.height))

        for (int yy = by0; yy < by1; yy++) {
            for (int xx = bx0; xx < bx1; xx++) {
                int v = tmpRaster.getSample(xx, yy, 0)
                if (v > 0) {
                    instanceRaster.setSample(xx, yy, 0, runningCellId)
                }
            }
        }

        appendTextFile(cells2labelsPath, "${classId}\n")

        String objName = cellObj.getName()
        if (objName == null)
            objName = ""

        double cxGlobal = roi.getCentroidX()
        double cyGlobal = roi.getCentroidY()
        double cxLocal = (cxGlobal - x0) / downsample
        double cyLocal = (cyGlobal - y0) / downsample

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

        runningCellId++

        if ((idx + 1) % 100 == 0) {
            print "Rasterized ${idx + 1}/${cells.size()} cells"
        }
    }

    ImageIO.write(instanceMask, "TIFF", new File(instanceMaskPath))

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

// =====================================
// WRITE TRAIN/VAL SPLIT + config.json
// =====================================

if (exportedImageIds.isEmpty()) {
    print "ERROR: No image IDs were exported, so config.json was not created."
    return
}

def shuffledIds = new ArrayList(exportedImageIds)
Collections.shuffle(shuffledIds, new Random(splitSeed))

int nTotal = shuffledIds.size()
int nVal = (int)Math.round(nTotal * valFraction)

if (nTotal >= 2) {
    nVal = Math.max(1, Math.min(nVal, nTotal - 1))
} else {
    nVal = 0
}

def valSet = shuffledIds.subList(0, nVal)
def trainSet = shuffledIds.subList(nVal, nTotal)

if (trainSet.isEmpty() && !valSet.isEmpty()) {
    trainSet = [valSet[0]]
    valSet = []
}

def hierarchyMatch = [:]
if (writeHierarchyMatch) {
    classToId.each { cls, id ->
        hierarchyMatch["${id}"] = writeHierarchyMatchFromClasses ? cls : "${id}"
    }
}

def configPath = buildFilePath(exportRoot, "config.json")

def jsonLines = []
jsonLines << "{"
jsonLines << "  \"crop_input_size\": ${cropInputSize},"
jsonLines << "  \"crop_size\": ${cropSize},"
jsonLines << "  \"root_dir\": " + jsonString(normalizeForJsonPath(exportRoot)) + ","
jsonLines << "  \"train_set\": " + jsonArrayOfStrings(trainSet) + ","
jsonLines << "  \"val_set\": " + jsonArrayOfStrings(valSet) + ","
jsonLines << "  \"num_classes\": ${classToId.size()},"
jsonLines << "  \"epoch_max\": ${epochMax},"
jsonLines << "  \"lr\": ${learningRate},"
jsonLines << "  \"to_pad\": ${toPad},"
jsonLines << "  \"blacklist\": " + jsonArrayOfStrings(blacklist) + ","
jsonLines << "  \"channels_path\": " + jsonString(normalizeForJsonPath(channelsPath)) + ","
jsonLines << "  \"weight_to_eval\": \"\","
jsonLines << "  \"sample_batch\": ${sampleBatch},"

jsonLines << "  \"hierarchy_match\": {"
def hmEntries = hierarchyMatch.collect { k, v ->
    "    " + jsonString(k.toString()) + ": " + jsonString(v.toString())
}
if (hmEntries.isEmpty()) {
    // no entries
} else {
    for (int i = 0; i < hmEntries.size(); i++) {
        if (i < hmEntries.size() - 1)
            jsonLines << hmEntries[i] + ","
        else
            jsonLines << hmEntries[i]
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

def trainSetPath = buildFilePath(exportRoot, "train_set.txt")
def valSetPath = buildFilePath(exportRoot, "val_set.txt")
writeTextFile(trainSetPath, trainSet.join("\n") + (trainSet.isEmpty() ? "" : "\n"))
writeTextFile(valSetPath, valSet.join("\n") + (valSet.isEmpty() ? "" : "\n"))

// =====================================
// DONE
// =====================================

print ""
print "====================================="
print "ALL DONE."
print "Export root: ${exportRoot}"
print "Images exported: ${exportedCount}"
print "Global class map: ${classMapPath}"
print "Channels file: ${channelsPath}"
print "Config file: ${configPath}"
print "Train set size: ${trainSet.size()}"
print "Val set size: ${valSet.size()}"