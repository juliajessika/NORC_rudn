import qupath.lib.regions.RegionRequest
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.images.servers.TransformedServerBuilder

// ==========================
// Settings
// ==========================
def imageData = getCurrentImageData()
def server = imageData.getServer()
def plane = ImagePlane.getDefaultPlane()
def hierarchy = imageData.getHierarchy()

def selected = getSelectedObject()
if (selected == null || selected.getROI() == null) {
    print "❌ No ROI selected!"
    return
}

def roi = selected.getROI()

int roiX = (int)Math.floor(roi.getBoundsX())
int roiY = (int)Math.floor(roi.getBoundsY())
int roiW = (int)Math.ceil(roi.getBoundsWidth())
int roiH = (int)Math.ceil(roi.getBoundsHeight())

int imgW = server.getWidth()
int imgH = server.getHeight()

print "ROI bounds: x=${roiX}, y=${roiY}, w=${roiW}, h=${roiH}"

// folders
def tileDir = buildFilePath(PROJECT_BASE_DIR, "tiles_roi")
mkdirs(tileDir)

// tile params
int tileSize = 512
int overlap = 64
int step = tileSize - overlap
double downsample = 1.0d

if (step <= 0) {
    print "❌ overlap must be smaller than tileSize"
    return
}

// ==========================
// Docker / model settings
// ==========================
// Build the image beforehand:
// docker build -t qupath-deepcell-seg:latest .

String dockerExe = "docker"
String dockerImage = "qupath-deepcell-seg:latest"

// Host folder containing the weights file
String weightsHostDir = "C:/Users/julia/OneDrive/Desktop/melanoma"

// Weight file name inside that folder
String weightsFileName = "nuclear_finetuned_best.weights.h5"

// DeepCell inference settings
String imageMpp = "0.65"
String maximaThreshold = "0.05"
String interiorThreshold = "0.3"
String minObjectArea = "40"
String minPolygonPoints = "6"
String simplifyEveryNth = "2"

// ==========================
// Find DAPI channel only
// ==========================
def metadata = server.getMetadata()
def channels = metadata.getChannels()

int dapiIndex = -1
for (int i = 0; i < channels.size(); i++) {
    def nm = channels.get(i).getName()
    if (nm != null && nm.toLowerCase().contains("dapi")) {
        dapiIndex = i
        break
    }
}

if (dapiIndex < 0) {
    dapiIndex = 0
    print "⚠️ DAPI channel not found by name, using channel 0: ${channels.get(dapiIndex).getName()}"
} else {
    print "✅ Using DAPI channel index ${dapiIndex}: ${channels.get(dapiIndex).getName()}"
}

def exportServer = new TransformedServerBuilder(server)
        .extractChannels(dapiIndex)
        .build()

// cleanup
def tileFolderFile = new File(tileDir)
if (tileFolderFile.exists()) {
    tileFolderFile.eachFile { f ->
        def name = f.name.toLowerCase()
        if (name.endsWith(".txt") || name.endsWith(".tif")) {
            f.delete()
        }
    }
}

def tileFiles = []
def tileInfo = [:]

// ==========================
// Helper: tile starts
// ==========================
def makeStarts = { int minCoord, int roiSize, int fullSize, int stepSize, int imageLimit ->
    def starts = []

    int start0 = Math.max(0, minCoord)
    int endExclusive = Math.min(imageLimit, minCoord + roiSize)

    if (roiSize <= 0 || endExclusive <= start0) {
        return starts
    }

    int s = start0
    while (s < endExclusive) {
        starts << s
        s += stepSize
    }

    int lastStart = Math.max(start0, endExclusive - fullSize)
    lastStart = Math.min(lastStart, Math.max(0, imageLimit - fullSize))

    if (!starts.contains(lastStart)) {
        starts << lastStart
    }

    starts = starts.unique().sort()
    return starts
}

def xStarts = makeStarts(roiX, roiW, tileSize, step, imgW)
def yStarts = makeStarts(roiY, roiH, tileSize, step, imgH)

if (xStarts.isEmpty() || yStarts.isEmpty()) {
    print "❌ Could not generate tiles for ROI"
    return
}

print "Tile grid: ${xStarts.size()} x ${yStarts.size()}"

// ==========================
// Export tiles
// ==========================
for (int iy = 0; iy < yStarts.size(); iy++) {
    int y = yStarts[iy] as int

    for (int ix = 0; ix < xStarts.size(); ix++) {
        int x = xStarts[ix] as int

        int w = Math.min(tileSize, imgW - x)
        int h = Math.min(tileSize, imgH - y)

        if (w <= 0 || h <= 0)
            continue

        if (!roi.intersects(x as double, y as double, w as double, h as double)) {
            continue
        }

        def request = RegionRequest.createInstance(
            exportServer.getPath(),
            downsample,
            x, y, w, h
        )

        def filename = buildFilePath(tileDir, String.format("tile_x%d_y%d_w%d_h%d.tif", x, y, w, h))
        writeImageRegion(exportServer, request, filename)
        tileFiles << filename

        tileInfo["${x}_${y}"] = [
            x  : x,
            y  : y,
            w  : w,
            h  : h,
            ix : ix,
            iy : iy,
            nX : xStarts.size(),
            nY : yStarts.size()
        ]
    }
}

print "✅ ROI tiling complete! Saved ${tileFiles.size()} tiles."

if (tileFiles.isEmpty()) {
    print "❌ No tiles exported."
    return
}

// ==========================
// Call Docker container
// ==========================
def tileDirAbs = new File(tileDir).getAbsolutePath()
def weightsHostDirAbs = new File(weightsHostDir).getAbsolutePath()

def weightsHostDirFile = new File(weightsHostDirAbs)
if (!weightsHostDirFile.exists() || !weightsHostDirFile.isDirectory()) {
    print "❌ Weights folder not found: ${weightsHostDirAbs}"
    return
}

def expectedWeights = new File(weightsHostDirFile, weightsFileName)
if (!expectedWeights.exists()) {
    print "❌ Weights file not found: ${expectedWeights.getAbsolutePath()}"
    return
}

List<String> cmd = [
    dockerExe,
    "run",
    "--rm",
    "-v",
    tileDirAbs + ":/data",
    "-v",
    weightsHostDirAbs + ":/model:ro",
    "-e",
    "TILE_DIR=/data",
    "-e",
    "WEIGHTS_PATH=/model/" + weightsFileName,
    "-e",
    "IMAGE_MPP=" + imageMpp,
    "-e",
    "MAXIMA_THRESHOLD=" + maximaThreshold,
    "-e",
    "INTERIOR_THRESHOLD=" + interiorThreshold,
    "-e",
    "MIN_OBJECT_AREA=" + minObjectArea,
    "-e",
    "MIN_POLYGON_POINTS=" + minPolygonPoints,
    "-e",
    "SIMPLIFY_EVERY_NTH=" + simplifyEveryNth,
    dockerImage
].collect { it.toString() }

println "Running command: " + cmd

def pb = new ProcessBuilder(cmd)
pb.redirectErrorStream(true)
def proc = pb.start()

proc.inputStream.eachLine { line ->
    println "[DOCKER] ${line}"
}

int exitCode = proc.waitFor()
println "Docker exit code: ${exitCode}"

if (exitCode != 0) {
    print "❌ Docker container failed with exit code ${exitCode}"
    return
}

print "✅ Container segmentation complete!"

// ==========================
// Read back polygons
// ==========================
def txtFiles = tileFolderFile.listFiles()?.findAll {
    it.name.toLowerCase().endsWith(".txt")
}

if (txtFiles == null || txtFiles.isEmpty()) {
    print "❌ No annotation txt files found in ${tileDir}"
    return
}

def childObjectsToAdd = []

txtFiles.each { file ->

    def matcher = (file.name =~ /tile_x(\d+)_y(\d+)(?:_w(\d+)_h(\d+))?\.txt/)
    if (!matcher.matches()) {
        print "⚠️ Skipping file with unexpected name: ${file.name}"
        return
    }

    int tileX = matcher[0][1] as int
    int tileY = matcher[0][2] as int

    def info = tileInfo["${tileX}_${tileY}"]
    if (info == null) {
        print "⚠️ Missing tile info for ${file.name}"
        return
    }

    int ix = info.ix as int
    int iy = info.iy as int
    int w = info.w as int
    int h = info.h as int

    double leftTrim = 0.0d
    double rightTrim = 0.0d
    double topTrim = 0.0d
    double bottomTrim = 0.0d

    if (ix > 0) {
        int prevX = xStarts[ix - 1] as int
        int overlapLeft = (prevX + tileSize) - tileX
        leftTrim = Math.max(0.0d, ((double)overlapLeft) / 2.0d)
    }

    if (ix < xStarts.size() - 1) {
        int nextX = xStarts[ix + 1] as int
        int overlapRight = (tileX + w) - nextX
        rightTrim = Math.max(0.0d, ((double)overlapRight) / 2.0d)
    }

    if (iy > 0) {
        int prevY = yStarts[iy - 1] as int
        int overlapTop = (prevY + tileSize) - tileY
        topTrim = Math.max(0.0d, ((double)overlapTop) / 2.0d)
    }

    if (iy < yStarts.size() - 1) {
        int nextY = yStarts[iy + 1] as int
        int overlapBottom = (tileY + h) - nextY
        bottomTrim = Math.max(0.0d, ((double)overlapBottom) / 2.0d)
    }

    double keepMinX = leftTrim
    double keepMaxX = ((double)w) - rightTrim
    double keepMinY = topTrim
    double keepMaxY = ((double)h) - bottomTrim

    file.eachLine { line ->
        line = line.trim()
        if (!line)
            return

        def pts = []
        def pairs = line.split(";")

        pairs.each { pair ->
            def xy = pair.trim().split(",")
            if (xy.size() != 2)
                return

            double localX = xy[0] as double
            double localY = xy[1] as double
            double globalX = tileX + localX
            double globalY = tileY + localY

            pts << [localX, localY, globalX, globalY]
        }

        if (pts.size() < 3)
            return

        double cxLocal = (pts.collect { it[0] as double }.sum() as double) / (double)pts.size()
        double cyLocal = (pts.collect { it[1] as double }.sum() as double) / (double)pts.size()
        double cxGlobal = (pts.collect { it[2] as double }.sum() as double) / (double)pts.size()
        double cyGlobal = (pts.collect { it[3] as double }.sum() as double) / (double)pts.size()

        if (cxLocal < keepMinX || cxLocal >= keepMaxX || cyLocal < keepMinY || cyLocal >= keepMaxY) {
            return
        }

        if (!roi.contains(cxGlobal, cyGlobal)) {
            return
        }

        double[] xs = pts.collect { it[2] as double } as double[]
        double[] ys = pts.collect { it[3] as double } as double[]

        def poly = ROIs.createPolygonROI(xs, ys, plane)

        def cellObj = PathObjects.createAnnotationObject(poly)
        cellObj.setName("Cell")

        childObjectsToAdd << cellObj
    }
}

// ==========================
// Add as CHILD objects under selected ROI
// ==========================
if (!childObjectsToAdd.isEmpty()) {

    /*
    def oldChildren = selected.getChildObjects() == null ? [] : new ArrayList(selected.getChildObjects())
    if (!oldChildren.isEmpty()) {
        hierarchy.removeObjects(oldChildren, true)
        print "🗑️ Removed ${oldChildren.size()} old child objects from selected ROI."
    }
    */

    selected.addChildObjects(childObjectsToAdd)
    hierarchy.fireHierarchyChangedEvent(this)

    print "✅ Added ${childObjectsToAdd.size()} child cell annotations under the selected parent ROI."
} else {
    print "⚠️ No valid polygons were loaded."
}