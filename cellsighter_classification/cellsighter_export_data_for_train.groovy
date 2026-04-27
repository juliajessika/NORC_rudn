/**
 * QuPath Groovy script
 *
 * YAML-driven export of CellSighter-style training data from parent annotations
 * that contain classified child cell objects.
 *
 * Expected YAML file:
 *   {PROJECT_BASE_DIR}/cell_sighter_export_config.yaml
 *
 * CellSighter expects:
 *   {root}/CellTypes/data/images/<image_id>.tiff
 *   {root}/CellTypes/cells/<image_id>.tiff
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
import java.util.LinkedHashSet

// ============================================================
// Minimal YAML parser
// Supports this config style:
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
// Generic helpers
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

def asLong = { def value, long defaultValue ->
    if (value == null)
        return defaultValue

    if (value instanceof Number)
        return value.longValue()

    return value.toString().trim().toLong()
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

// ============================================================
// Load YAML config
// ============================================================

def configFile = new File(buildFilePath(PROJECT_BASE_DIR, "cellsighter_export_config.yaml"))

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

// Export settings
double downsample = asDouble(getCfg(cfg, ["export", "downsample"], 1.0), 1.0)

String exportRootName = getCfg(cfg, ["export", "export_root_name"], "cellsighter_export").toString()

String rawExt = ensureExtension(getCfg(cfg, ["export", "raw_ext"], ".tiff")?.toString(), ".tiff")
String maskExt = ensureExtension(getCfg(cfg, ["export", "mask_ext"], ".tiff")?.toString(), ".tiff")

boolean exportSelectedAnnotationsOnly = asBool(getCfg(cfg, ["export", "selected_annotations_only"], false), false)
boolean skipUnclassifiedCells = asBool(getCfg(cfg, ["export", "skip_unclassified_cells"], true), true)
boolean simplifyClassNames = asBool(getCfg(cfg, ["export", "simplify_class_names"], true), true)

double minCellAreaPx = asDouble(getCfg(cfg, ["export", "min_cell_area_px"], 0.0), 0.0)

boolean useRecursiveChildren = asBool(getCfg(cfg, ["export", "use_recursive_children"], true), true)
boolean prependBackgroundLabelInTxt = asBool(getCfg(cfg, ["export", "prepend_background_label_in_txt"], true), true)
boolean skipEmptyAnnotations = asBool(getCfg(cfg, ["export", "skip_empty_annotations"], true), true)

String imageIdPrefix = getCfg(cfg, ["export", "image_id_prefix"], "FOV").toString()

// Split settings
double valFraction = asDouble(getCfg(cfg, ["split", "val_fraction"], 0.5), 0.5)
long splitSeed = asLong(getCfg(cfg, ["split", "seed"], 12345L), 12345L)

// Training config settings
int cropInputSize = asInt(getCfg(cfg, ["training_config", "crop_input_size"], 60), 60)
int cropSize = asInt(getCfg(cfg, ["training_config", "crop_size"], 128), 128)
int epochMax = asInt(getCfg(cfg, ["training_config", "epoch_max"], 50), 50)

double learningRate = asDouble(getCfg(cfg, ["training_config", "learning_rate"], 0.001), 0.001)

boolean toPad = asBool(getCfg(cfg, ["training_config", "to_pad"], false), false)
boolean sampleBatch = asBool(getCfg(cfg, ["training_config", "sample_batch"], true), true)
boolean aug = asBool(getCfg(cfg, ["training_config", "aug"], true), true)

Integer sizeData = asNullableInteger(getCfg(cfg, ["training_config", "size_data"], null))

int batchSize = asInt(getCfg(cfg, ["training_config", "batch_size"], 32), 32)
int numWorkers = asInt(getCfg(cfg, ["training_config", "num_workers"], 0), 0)

def blacklist = asList(getCfg(cfg, ["training_config", "blacklist"], []))

// Hierarchy settings
boolean writeHierarchyMatch = asBool(getCfg(cfg, ["hierarchy", "write_hierarchy_match"], true), true)
boolean writeHierarchyMatchFromClasses = asBool(getCfg(cfg, ["hierarchy", "write_hierarchy_match_from_classes"], true), true)

if (downsample <= 0) {
    println "ERROR: export.downsample must be > 0"
    return
}

if (valFraction < 0 || valFraction > 1) {
    println "ERROR: split.val_fraction must be between 0 and 1"
    return
}

println ""
println "Settings:"
println "  downsample: ${downsample}"
println "  export_root_name: ${exportRootName}"
println "  selected_annotations_only: ${exportSelectedAnnotationsOnly}"
println "  skip_unclassified_cells: ${skipUnclassifiedCells}"
println "  use_recursive_children: ${useRecursiveChildren}"
println "  val_fraction: ${valFraction}"

// ============================================================
// Script helpers
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

// ============================================================
// Initial setup
// ============================================================

def imageData = getCurrentImageData()
def server = imageData.getServer()
def imageName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

if (getProject() == null) {
    println "ERROR: Please open a QuPath project first."
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

// ============================================================
// First pass: collect global class names
// ============================================================

def globalClassNames = new LinkedHashSet<String>()
def annotationToCells = [:]

parentAnnotations.eachWithIndex { parentAnn, annIdx ->
    def cells = getCellsForParent(parentAnn)
    annotationToCells[parentAnn] = cells

    cells.each { obj ->
        def cls = cleanClassName(obj.getPathClass())
        if (cls != null && cls.trim().length() > 0)
            globalClassNames << cls
    }
}

if (globalClassNames.isEmpty()) {
    println "ERROR: No classified cells found in selected annotations."
    return
}

def sortedClassNames = globalClassNames.toList().sort()
def classToId = [:]

sortedClassNames.eachWithIndex { cls, idx ->
    classToId[cls] = idx
}

println ""
println "Global class mapping:"
classToId.each { k, v ->
    println "  ${k} -> ${v}"
}

// Save global class map
def classMapPath = buildFilePath(exportRoot, "class_map.csv")
writeTextFile(classMapPath, "class_name,class_id\n")

classToId.each { cls, clsId ->
    appendTextFile(classMapPath, "${csvEscape(cls)},${clsId}\n")
}

// Write channels.txt from server metadata
def channelsPath = buildFilePath(exportRoot, "channels.txt")
def channelsText = new StringBuilder()

try {
    def metadata = server.getMetadata()
    def channels = metadata.getChannels()

    if (channels != null && !channels.isEmpty()) {
        channels.each { ch ->
            def nm = ch.getName()

            if (nm == null || nm.trim().length() == 0)
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

writeTextFile(
        exportSummaryPath,
        "image_id,source_image,parent_annotation,annotation_index,cell_count,raw_path,mask_path,labels_path\n"
)

// Keep exported image IDs for config generation
def exportedImageIds = []

// ============================================================
// Second pass: export each annotation as one image ID
// ============================================================

int exportedCount = 0

parentAnnotations.eachWithIndex { parentAnn, annIdx ->

    def parentROI = parentAnn.getROI()

    String parentLabel = parentAnn.getName()
    if (parentLabel == null || parentLabel.trim().length() == 0)
        parentLabel = "annotation_${annIdx + 1}"

    def cells = annotationToCells[parentAnn]
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

    String imageId = safeName("${imageIdPrefix}_${imageName}_${annIdx + 1}_${parentLabel}")

    println ""
    println "====================================="
    println "Exporting annotation ${annIdx + 1}/${parentAnnotations.size()}: ${parentLabel}"
    println "Image ID: ${imageId}"
    println "Cells kept: ${cells.size()}"

    int x0 = (int)Math.floor(parentROI.getBoundsX())
    int y0 = (int)Math.floor(parentROI.getBoundsY())
    int w = (int)Math.ceil(parentROI.getBoundsWidth())
    int h = (int)Math.ceil(parentROI.getBoundsHeight())

    int outW = (int)Math.ceil(w / downsample)
    int outH = (int)Math.ceil(h / downsample)

    if (outW <= 0 || outH <= 0) {
        println "WARNING: annotation '${parentLabel}' produced non-positive output size. Skipping."
        return
    }

    println "Region: x=${x0}, y=${y0}, w=${w}, h=${h}, downsample=${downsample}"
    println "Output size: ${outW} x ${outH}"

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

    writeTextFile(
            cellTablePath,
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

        runningCellId++

        if ((idx + 1) % 100 == 0) {
            println "Rasterized ${idx + 1}/${cells.size()} cells"
        }
    }

    ImageIO.write(instanceMask, "TIFF", new File(instanceMaskPath))

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
// Write train/val split + config.json
// ============================================================

if (exportedImageIds.isEmpty()) {
    println "ERROR: No image IDs were exported, so config.json was not created."
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

if (!hmEntries.isEmpty()) {
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

// ============================================================
// Done
// ============================================================

println ""
println "====================================="
println "ALL DONE."
println "Export root: ${exportRoot}"
println "Images exported: ${exportedCount}"
println "Global class map: ${classMapPath}"
println "Channels file: ${channelsPath}"
println "Config file: ${configPath}"
println "Train set size: ${trainSet.size()}"
println "Val set size: ${valSet.size()}"
