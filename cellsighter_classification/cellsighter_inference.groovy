/**
 * QuPath Groovy script
 *
 * YAML-driven CellSighter inference workflow:
 * 1. Export CellSighter-style data from current annotations.
 * 2. Run CellSighter evaluation with an already-trained model via conda.
 * 3. Import predictions back into QuPath cell objects.
 *
 * Expected YAML file:
 *   {PROJECT_BASE_DIR}/cell_sighter_inference_config.yaml
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
 *   instead of relying on list index cell_id - 1.
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
// Minimal YAML parser
// Supports:
//   key: value
//   key:
//     nested_key: value
//   list:
//     - item
//     - "item"
//   inline_list: [1, "DAPI", true]
// Does NOT support anchors, multiline strings, lists of maps, etc.
// ============================================================

def stripYamlComment = { String s ->
    boolean inSingle = false
    boolean inDouble = false
    def out = new StringBuilder()

    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i)

        if (c == (char)39 && !inDouble) {
            inSingle = !inSingle
            out.append(c)
            continue
        }

        if (c == (char)34 && !inSingle) {
            inDouble = !inDouble
            out.append(c)
            continue
        }

        if (c == (char)35 && !inSingle && !inDouble) {
            break
        }

        out.append(c)
    }

    return out.toString()
}

def splitInlineList = { String s ->
    def parts = []
    def cur = new StringBuilder()
    boolean inSingle = false
    boolean inDouble = false

    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i)

        if (c == (char)39 && !inDouble) {
            inSingle = !inSingle
            cur.append(c)
            continue
        }

        if (c == (char)34 && !inSingle) {
            inDouble = !inDouble
            cur.append(c)
            continue
        }

        if (c == (char)44 && !inSingle && !inDouble) {
            parts << cur.toString().trim()
            cur.setLength(0)
            continue
        }

        cur.append(c)
    }

    def last = cur.toString().trim()
    if (last.length() > 0)
        parts << last

    return parts
}

def parseScalar
parseScalar = { String s ->
    if (s == null)
        return null

    s = s.trim()

    if (s.length() == 0)
        return ""

    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
        return s.substring(1, s.length() - 1)
    }

    if (s.equalsIgnoreCase("true") || s.equalsIgnoreCase("yes") || s.equalsIgnoreCase("on"))
        return true

    if (s.equalsIgnoreCase("false") || s.equalsIgnoreCase("no") || s.equalsIgnoreCase("off"))
        return false

    if (s.equalsIgnoreCase("null") || s.equalsIgnoreCase("none"))
        return null

    if (s.startsWith("[") && s.endsWith("]")) {
        def inside = s.substring(1, s.length() - 1).trim()
        if (inside.length() == 0)
            return []
        return splitInlineList(inside).collect { parseScalar(it) }
    }

    if (s ==~ /[-+]?\d+/) {
        try {
            return Integer.parseInt(s)
        } catch (Throwable ignored) {
            return Long.parseLong(s)
        }
    }

    if (s ==~ /[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?/) {
        return Double.parseDouble(s)
    }

    if (s ==~ /[-+]?\d+[eE][-+]?\d+/) {
        return Double.parseDouble(s)
    }

    return s
}

def parseSimpleYaml = { String text ->
    def records = []

    int lineNo = 0
    text.eachLine { raw ->
        lineNo++

        def noComment = stripYamlComment(raw)
        if (noComment.trim().length() == 0)
            return

        int indent = 0
        while (indent < noComment.length() && noComment.charAt(indent) == (char)32) {
            indent++
        }

        if (indent < noComment.length() && noComment.charAt(indent) == (char)9) {
            throw new RuntimeException("Tabs are not supported in YAML indentation, line ${lineNo}")
        }

        records << [
                indent: indent,
                text  : noComment.trim(),
                lineNo: lineNo
        ]
    }

    def root = [:]
    def stack = [[indent: -1, obj: root]]

    for (int i = 0; i < records.size(); i++) {
        def rec = records[i]
        int indent = rec.indent as int
        String line = rec.text as String

        while (stack.size() > 1 && indent <= (stack[-1].indent as int)) {
            stack.remove(stack.size() - 1)
        }

        def parent = stack[-1].obj

        if (line.startsWith("- ")) {
            if (!(parent instanceof List)) {
                throw new RuntimeException("List item found outside list at YAML line ${rec.lineNo}: ${line}")
            }

            String item = line.substring(2).trim()
            parent << parseScalar(item)
            continue
        }

        int colon = line.indexOf(":")
        if (colon < 0) {
            throw new RuntimeException("Expected key: value at YAML line ${rec.lineNo}: ${line}")
        }

        String key = line.substring(0, colon).trim()
        String rest = line.substring(colon + 1).trim()

        if (!(parent instanceof Map)) {
            throw new RuntimeException("Key-value found inside list at YAML line ${rec.lineNo}: ${line}")
        }

        if (rest.length() == 0) {
            def next = i + 1 < records.size() ? records[i + 1] : null
            def child

            if (next != null && (next.indent as int) > indent && (next.text as String).startsWith("- ")) {
                child = []
            } else {
                child = [:]
            }

            parent[key] = child
            stack << [indent: indent, obj: child]
        } else {
            parent[key] = parseScalar(rest)
        }
    }

    return root
}

// ============================================================
// Generic config helpers
// ============================================================

def getCfg = { Map root, List path, defaultValue = null ->
    def cur = root

    for (def key in path) {
        if (cur == null)
            return defaultValue

        if (cur instanceof Map && cur.containsKey(key)) {
            cur = cur[key]
        } else {
            return defaultValue
        }
    }

    return cur == null ? defaultValue : cur
}

def asBool = { def value, boolean defaultValue = false ->
    if (value == null)
        return defaultValue

    if (value instanceof Boolean)
        return value

    def s = value.toString().trim().toLowerCase()

    if (s in ["true", "yes", "y", "1", "on"])
        return true

    if (s in ["false", "no", "n", "0", "off"])
        return false

    return defaultValue
}

def asInt = { def value, int defaultValue ->
    if (value == null)
        return defaultValue

    if (value instanceof Number)
        return value.intValue()

    return value.toString().trim().toInteger()
}

def asDouble = { def value, double defaultValue ->
    if (value == null)
        return defaultValue

    if (value instanceof Number)
        return value.doubleValue()

    return value.toString().trim().toDouble()
}

def asNullableInteger = { def value ->
    if (value == null)
        return null

    if (value instanceof Number)
        return value.intValue()

    def s = value.toString().trim()
    if (s.length() == 0 || s.equalsIgnoreCase("null") || s.equalsIgnoreCase("none"))
        return null

    return s.toInteger()
}

def asList = { def value ->
    if (value == null)
        return []
    if (value instanceof List)
        return value
    return [value]
}

def ensureExtension = { String ext, String defaultExt ->
    if (ext == null || ext.trim().length() == 0)
        return defaultExt

    ext = ext.trim()

    if (!ext.startsWith("."))
        ext = "." + ext

    return ext
}

def optionalString = { def value ->
    if (value == null)
        return null

    String s = value.toString().trim()
    if (s.length() == 0 || s.equalsIgnoreCase("null") || s.equalsIgnoreCase("none"))
        return null

    return s
}

// ============================================================
// Load YAML config
// ============================================================

def configFile = new File(buildFilePath(PROJECT_BASE_DIR, "cell_sighter_inference_config.yaml"))

if (!configFile.exists()) {
    println "ERROR: YAML config not found:"
    println "  ${configFile.getAbsolutePath()}"
    return
}

def cfg

try {
    cfg = parseSimpleYaml(configFile.getText("UTF-8")) as Map
} catch (Throwable e) {
    println "ERROR: Failed to parse YAML config:"
    println e.getMessage()
    return
}

println "Loaded YAML config:"
println "  ${configFile.getAbsolutePath()}"

// ============================================================
// Settings from YAML
// ============================================================

double downsample = asDouble(getCfg(cfg, ["export", "downsample"], 1.0), 1.0)

String exportRootName = getCfg(cfg, ["export", "export_root_name"], "cellsighter_export_inference1").toString()
String rawExt = ensureExtension(getCfg(cfg, ["export", "raw_ext"], ".tiff")?.toString(), ".tiff")
String maskExt = ensureExtension(getCfg(cfg, ["export", "mask_ext"], ".tiff")?.toString(), ".tiff")

boolean exportSelectedAnnotationsOnly = asBool(getCfg(cfg, ["export", "selected_annotations_only"], false), false)
boolean skipUnclassifiedCells = asBool(getCfg(cfg, ["export", "skip_unclassified_cells"], false), false)
boolean simplifyClassNames = asBool(getCfg(cfg, ["export", "simplify_class_names"], true), true)
double minCellAreaPx = asDouble(getCfg(cfg, ["export", "min_cell_area_px"], 0.0), 0.0)
boolean useRecursiveChildren = asBool(getCfg(cfg, ["export", "use_recursive_children"], true), true)
boolean prependBackgroundLabelInTxt = asBool(getCfg(cfg, ["export", "prepend_background_label_in_txt"], true), true)
boolean skipEmptyAnnotations = asBool(getCfg(cfg, ["export", "skip_empty_annotations"], true), true)

String imageIdPrefix = getCfg(cfg, ["export", "image_id_prefix"], "FOV").toString()

boolean maskRawOutsideAnnotation = asBool(getCfg(cfg, ["export", "mask_raw_outside_annotation"], false), false)
boolean exportNativeRaw = asBool(getCfg(cfg, ["export", "export_native_raw"], true), true)

int cropInputSize = asInt(getCfg(cfg, ["training_config", "crop_input_size"], 60), 60)
int cropSize = asInt(getCfg(cfg, ["training_config", "crop_size"], 128), 128)
int epochMax = asInt(getCfg(cfg, ["training_config", "epoch_max"], 50), 50)
double learningRate = asDouble(getCfg(cfg, ["training_config", "learning_rate"], 0.001), 0.001)
boolean toPad = asBool(getCfg(cfg, ["training_config", "to_pad"], true), true)
boolean sampleBatch = asBool(getCfg(cfg, ["training_config", "sample_batch"], true), true)
boolean aug = asBool(getCfg(cfg, ["training_config", "aug"], false), false)
Integer sizeData = asNullableInteger(getCfg(cfg, ["training_config", "size_data"], null))
int batchSize = asInt(getCfg(cfg, ["training_config", "batch_size"], 32), 32)
int numWorkers = asInt(getCfg(cfg, ["training_config", "num_workers"], 0), 0)
def blacklist = asList(getCfg(cfg, ["training_config", "blacklist"], []))

boolean writeHierarchyMatch = asBool(getCfg(cfg, ["hierarchy", "write_hierarchy_match"], true), true)
boolean writeHierarchyMatchFromClasses = asBool(getCfg(cfg, ["hierarchy", "write_hierarchy_match_from_classes"], true), true)

boolean runCellSighterEvaluation = asBool(getCfg(cfg, ["cellsighter", "run_evaluation"], true), true)

String cellSighterEvalPy = getCfg(cfg, ["cellsighter", "eval_py"], "").toString()
String condaExe = getCfg(cfg, ["cellsighter", "conda_exe"], "").toString()
String condaEnvName = getCfg(cfg, ["cellsighter", "conda_env_name"], "cellsighter").toString()
String trainedModelPath = getCfg(cfg, ["cellsighter", "trained_model_path"], "").toString()
String trainedClassMapCsv = getCfg(cfg, ["cellsighter", "trained_class_map_csv"], "").toString()
String externalResultsCsvPath = optionalString(getCfg(cfg, ["cellsighter", "results_csv"], null))

boolean writeAllMinusOneLabelsForEval = asBool(getCfg(cfg, ["labels", "write_all_minus_one_labels_for_eval"], true), true)

boolean writePredictedPathClass = asBool(getCfg(cfg, ["import_predictions", "write_predicted_path_class"], true), true)
boolean writePredictionMeasurements = asBool(getCfg(cfg, ["import_predictions", "write_prediction_measurements"], true), true)
boolean prefixImportedClassNames = asBool(getCfg(cfg, ["import_predictions", "prefix_imported_class_names"], false), false)

if (downsample <= 0) {
    println "ERROR: export.downsample must be > 0"
    return
}

if (trainedClassMapCsv == null || trainedClassMapCsv.trim().length() == 0) {
    println "ERROR: cellsighter.trained_class_map_csv is empty."
    return
}

if (trainedModelPath == null || trainedModelPath.trim().length() == 0) {
    println "ERROR: cellsighter.trained_model_path is empty."
    return
}

if (runCellSighterEvaluation) {
    if (cellSighterEvalPy == null || cellSighterEvalPy.trim().length() == 0) {
        println "ERROR: cellsighter.eval_py is empty."
        return
    }

    if (condaExe == null || condaExe.trim().length() == 0) {
        println "ERROR: cellsighter.conda_exe is empty."
        return
    }
}

println ""
println "Settings:"
println "  export_root_name: ${exportRootName}"
println "  downsample: ${downsample}"
println "  selected_annotations_only: ${exportSelectedAnnotationsOnly}"
println "  export_native_raw: ${exportNativeRaw}"
println "  run_evaluation: ${runCellSighterEvaluation}"
println "  trained_model_path: ${trainedModelPath}"
println "  trained_class_map_csv: ${trainedClassMapCsv}"

// ============================================================
// Helpers
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
    if (s == null || s.trim().length() == 0)
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

    def cells = childObjects.findAll { obj ->
        isUsableCellLikeObject(obj)
    }

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

def getChannelNames = { serverObj ->
    def out = []

    try {
        def metadata = serverObj.getMetadata()
        def channels = metadata?.getChannels()

        if (channels != null && !channels.isEmpty()) {
            channels.eachWithIndex { ch, i ->
                String nm = ch?.getName()

                if (nm == null || nm.trim().length() == 0)
                    nm = "Channel_${i + 1}"

                out << nm
            }
        }
    } catch (Exception e) {
    }

    if (out.isEmpty()) {
        try {
            int n = serverObj.nChannels()
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

        if (c == (char)34) {
            if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == (char)34) {
                sb.append((char)34)
                i++
            } else {
                inQuotes = !inQuotes
            }
        } else if (c == (char)44 && !inQuotes) {
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
        if (line == null || line.trim().length() == 0)
            return

        def cols = parseSimpleCsvLine(line)

        if (cols.size() < 2)
            return

        String className = cols[0]
        int classId = cols[1].trim().toInteger()

        idToClass[classId] = className
        classToId[className] = classId
    }

    return [
            idToClass: idToClass,
            classToId: classToId
    ]
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

    return [
            exitCode: exitCode,
            output  : lines.join("\n")
    ]
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

    candidates = candidates.sort { a, b ->
        b.lastModified() <=> a.lastModified()
    }

    return candidates[0]
}

// Parse string like "[0.1, 0.2, 0.7]" into List<Double>
def parseProbList = { String s ->
    def out = []

    if (s == null)
        return out

    String t = s.trim()

    if (t.length() == 0)
        return out

    if (t.startsWith("["))
        t = t.substring(1)

    if (t.endsWith("]"))
        t = t.substring(0, t.length() - 1)

    if (t.trim().length() == 0)
        return out

    t.split(",").each { part ->
        String p = part.trim()

        if (p.length() > 0) {
            try {
                out << Double.parseDouble(p)
            } catch (Exception e) {
            }
        }
    }

    return out
}

// ============================================================
// Initial setup
// ============================================================

def imageData = getCurrentImageData()
def server = imageData.getServer()
def hierarchy = imageData.getHierarchy()
def imageName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

if (getProject() == null) {
    println "ERROR: Please open a QuPath project first."
    return
}

def channelNames = getChannelNames(server)

def mapInfo

try {
    mapInfo = readClassMapCsv(trainedClassMapCsv)
} catch (Throwable e) {
    println "ERROR: Failed to read trained class map:"
    println e.getMessage()
    return
}

def trainedIdToClass = mapInfo.idToClass
def trainedClassToId = mapInfo.classToId

println "Loaded training class map from: ${trainedClassMapCsv}"

trainedIdToClass.keySet().sort().each { id ->
    println "  ${id} -> ${trainedIdToClass[id]}"
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
// Get parent annotations
// ============================================================

def parentAnnotations

if (exportSelectedAnnotationsOnly) {
    parentAnnotations = getSelectedObjects().findAll {
        it.isAnnotation() && it.getROI() != null && it.getROI().isArea()
    }
} else {
    parentAnnotations = getAnnotationObjects().findAll {
        it.getROI() != null && it.getROI().isArea()
    }
}

if (parentAnnotations.isEmpty()) {
    println "ERROR: No parent annotations found."
    return
}

println "Found ${parentAnnotations.size()} parent annotation(s) to export."

def imageIdToCellMap = [:]

def channelsPath = buildFilePath(exportRoot, "channels.txt")
writeTextFile(channelsPath, channelNames.join("\n") + "\n")

def classMapPath = buildFilePath(exportRoot, "class_map.csv")
writeTextFile(classMapPath, "class_name,class_id\n")

trainedIdToClass.keySet().sort().each { id ->
    appendTextFile(classMapPath, csvEscape(trainedIdToClass[id]) + "," + id + "\n")
}

def exportSummaryPath = buildFilePath(exportRoot, "export_summary.csv")

writeTextFile(
        exportSummaryPath,
        "image_id,source_image,parent_annotation,annotation_index,cell_count,raw_path,mask_path,labels_path\n"
)

def exportedImageIds = []
int exportedCount = 0

// ============================================================
// Export each annotation
// ============================================================

parentAnnotations.eachWithIndex { parentAnn, annIdx ->

    def parentROI = parentAnn.getROI()

    String parentLabel = parentAnn.getName()
    if (parentLabel == null || parentLabel.trim().length() == 0)
        parentLabel = "annotation_${annIdx + 1}"

    def cells = getCellsForParent(parentAnn)

    if (cells == null)
        cells = []

    if (cells.isEmpty()) {
        String msg = "WARNING: annotation '${parentLabel}' has no suitable child cell objects."

        if (skipEmptyAnnotations) {
            println msg + " Skipping."
            return
        } else {
            println msg + " Exporting empty label/mask files."
        }
    }

    String imageId = safeName("${imageIdPrefix}_${imageName}_${parentLabel}")
    imageIdToCellMap[imageId] = [:]

    cleanupPreviousSampleFiles(imagesDir, cellsDir, cells2labelsDir, metaDir, imageId)

    println ""
    println "====================================="
    println "Exporting annotation ${annIdx + 1}/${parentAnnotations.size()}: ${parentLabel}"
    println "Image ID: ${imageId}"
    println "Cells kept: ${cells.size()}"

    int x0 = (int)Math.floor(parentROI.getBoundsX())
    int y0 = (int)Math.floor(parentROI.getBoundsY())
    int w = (int)Math.ceil(parentROI.getBoundsWidth())
    int h = (int)Math.ceil(parentROI.getBoundsHeight())

    if (w <= 0 || h <= 0) {
        println "WARNING: annotation '${parentLabel}' produced non-positive source size. Skipping."
        return
    }

    println "Region: x=${x0}, y=${y0}, w=${w}, h=${h}, downsample=${downsample}"

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
        println "ERROR: failed to read preview image for annotation '${parentLabel}': ${e.getMessage()}"
        return
    }

    if (previewBuffered == null) {
        println "ERROR: server.readBufferedImage(...) returned null for annotation '${parentLabel}'."
        return
    }

    outW = previewBuffered.getWidth()
    outH = previewBuffered.getHeight()

    if (outW <= 0 || outH <= 0) {
        println "ERROR: exported raw image has invalid size ${outW} x ${outH}."
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
                println "ERROR: failed to write raw TIFF: ${rawPath}"
                return
            }
        }
    } catch (Exception e) {
        println "ERROR: failed to export raw image for annotation '${parentLabel}': ${e.getMessage()}"
        return
    }

    if (!new File(rawPath).exists()) {
        println "ERROR: raw file was not created: ${rawPath}"
        return
    }

    println "Actual exported raw size used for mask: ${outW} x ${outH}"

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

    writeTextFile(
            cellTablePath,
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

        appendTextFile(
                cellTablePath,
                [
                        runningCellId,
                        classId,
                        csvEscape(className == null ? "" : className),
                        cxGlobal,
                        cyGlobal,
                        cxLocal,
                        cyLocal,
                        csvEscape(objName)
                ].join(",") + "\n"
        )

        imageIdToCellMap[imageId][runningCellId] = cellObj

        runningCellId++

        if ((idx + 1) % 100 == 0) {
            println "Rasterized ${idx + 1}/${cells.size()} cells"
        }
    }

    boolean maskOk = ImageIO.write(instanceMask, "TIFF", new File(instanceMaskPath))

    if (!maskOk || !new File(instanceMaskPath).exists()) {
        println "ERROR: failed to write instance mask TIFF: ${instanceMaskPath}"
        return
    }

    appendTextFile(
            exportSummaryPath,
            [
                    csvEscape(imageId),
                    csvEscape(imageName),
                    csvEscape(parentLabel),
                    annIdx + 1,
                    cells.size(),
                    csvEscape(rawPath),
                    csvEscape(instanceMaskPath),
                    csvEscape(cells2labelsPath)
            ].join(",") + "\n"
    )

    exportedImageIds << imageId

    println "Saved:"
    println "  raw image     -> ${rawPath}"
    println "  instance mask -> ${instanceMaskPath}"
    println "  labels txt    -> ${cells2labelsPath}"
    println "  cell table    -> ${cellTablePath}"

    exportedCount++
}

// ============================================================
// Write eval config.json
// ============================================================

if (exportedImageIds.isEmpty()) {
    println "ERROR: No image IDs were exported, so config.json was not created."
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

println ""
println "Config written: ${configPath}"

// ============================================================
// Run CellSighter eval through conda
// ============================================================

File resultsCsv = null

if (runCellSighterEvaluation) {
    if (!new File(condaExe).exists()) {
        println "ERROR: conda executable not found:"
        println "  ${condaExe}"
        return
    }

    if (!new File(cellSighterEvalPy).exists()) {
        println "ERROR: CellSighter eval.py not found:"
        println "  ${cellSighterEvalPy}"
        return
    }

    if (!new File(trainedModelPath).exists()) {
        println "ERROR: trained model file not found:"
        println "  ${trainedModelPath}"
        return
    }

    def cmd = [
            condaExe,
            "run",
            "-n",
            condaEnvName,
            "python",
            cellSighterEvalPy,
            "--base_path=" + new File(exportRoot).getAbsolutePath()
    ]

    def runInfo = runCommandAndCapture(cmd, new File(exportRoot))

    if (runInfo.exitCode != 0) {
        println "ERROR: CellSighter eval failed with exit code ${runInfo.exitCode}"
        return
    }

    resultsCsv = findValResultsFile(exportRoot)

    if (resultsCsv == null || !resultsCsv.exists()) {
        println "ERROR: Could not find val_results output inside:"
        println "  ${exportRoot}"
        return
    }

    println "Found CellSighter results: ${resultsCsv.getAbsolutePath()}"

} else {
    if (externalResultsCsvPath != null) {
        resultsCsv = new File(externalResultsCsvPath)
    } else {
        resultsCsv = new File(buildFilePath(exportRoot, "val_results.csv"))
    }

    if (!resultsCsv.exists()) {
        println "ERROR: run_evaluation=false, but results CSV was not found:"
        println "  ${resultsCsv.getAbsolutePath()}"
        return
    }

    println "Using existing CellSighter results:"
    println "  ${resultsCsv.getAbsolutePath()}"
}

// ============================================================
// Import predictions back into QuPath
// ============================================================

def resultLines = resultsCsv.readLines()

if (resultLines.size() < 2) {
    println "ERROR: Results file is empty or has no data rows:"
    println "  ${resultsCsv.getAbsolutePath()}"
    return
}

def header = parseSimpleCsvLine(resultLines[0])
def colIndex = [:]

header.eachWithIndex { c, i ->
    colIndex[c] = i
}

def requiredCols = ["pred", "pred_prob", "cell_id", "image_id"]
def missingRequiredCols = requiredCols.findAll {
    !colIndex.containsKey(it)
}

if (!missingRequiredCols.isEmpty()) {
    println "ERROR: Results CSV missing required columns: ${missingRequiredCols}"
    println "Header: ${header}"
    return
}

int importedCount = 0
int missingImageCount = 0
int missingCellCount = 0
int parseErrorCount = 0
int unknownPredClassCount = 0

resultLines.drop(1).eachWithIndex { line, lineIdx ->
    if (line == null || line.trim().length() == 0)
        return

    def cols = parseSimpleCsvLine(line)

    if (cols.size() < header.size()) {
        println "WARNING: skipping malformed results row ${lineIdx + 2}"
        return
    }

    String imageId = cols[colIndex["image_id"]]
    String predStr = cols[colIndex["pred"]]
    String predProbStr = cols[colIndex["pred_prob"]]
    String cellIdStr = cols[colIndex["cell_id"]]
    String probListStr = colIndex.containsKey("prob_list") ? cols[colIndex["prob_list"]] : ""

    if (imageId == null || imageId.length() == 0) {
        println "WARNING: result row ${lineIdx + 2} has empty image_id"
        return
    }

    def cellMapForImage = imageIdToCellMap[imageId]

    if (cellMapForImage == null) {
        missingImageCount++
        println "WARNING: no exported cell map for image_id='${imageId}'"
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
        println "WARNING: could not parse numeric values in results row ${lineIdx + 2}"
        return
    }

    def cellObj = cellMapForImage[cellId]

    if (cellObj == null) {
        missingCellCount++
        println "WARNING: cell_id=${cellId} not found for image_id='${imageId}'"
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
// Done
// ============================================================

println ""
println "====================================="
println "ALL DONE."
println "Export root: ${exportRoot}"
println "Images exported: ${exportedCount}"
println "Channels file: ${channelsPath}"
println "Config file: ${configPath}"
println "Results file: ${resultsCsv.getAbsolutePath()}"
println "Predictions imported into QuPath: ${importedCount}"
println "Missing image rows: ${missingImageCount}"
println "Missing cell rows: ${missingCellCount}"
println "Parse error rows: ${parseErrorCount}"
println "Unknown predicted class ids: ${unknownPredClassCount}"
println "Raw export mode: " + (exportNativeRaw ? "native channel-wise TIFF via writeImageRegion(...)" : "rendered RGB fallback")
