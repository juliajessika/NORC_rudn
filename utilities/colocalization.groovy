// --- USER CONFIGURATION SECTION (Edit these 5 lines!) ---
int FIRST_CHANNEL = 3
int SECOND_CHANNEL = 4
String objectType = "cell"   // "cell", "nucleus", "detection"
double ch1Background = 800
double ch2Background = 800
// -------------------------------------------------------

import qupath.lib.regions.RegionRequest
import static qupath.lib.gui.scripting.QPEx.*

def imageData = getCurrentImageData()
def server = imageData.getServer()
def hierarchy = imageData.getHierarchy()

// Get objects
def objectsToMeasure = []
switch (objectType) {
    case "cell":
        objectsToMeasure = getCellObjects()
        break
    case "nucleus":
        objectsToMeasure = getCellObjects().collect { it.getNucleus() }.findAll { it != null }
        break
    case "detection":
        objectsToMeasure = getDetectionObjects()
        break
    default:
        print "Unknown object type: ${objectType}"
        return
}

print "Processing ${objectsToMeasure.size()} objects"

// PCC with zero-variance protection
double calculatePCC(double[] x, double[] y) {
    int n = x.length
    if (n < 2) return Double.NaN

    double meanX = x.sum() / n
    double meanY = y.sum() / n

    double num = 0, denX = 0, denY = 0
    for (int i = 0; i < n; i++) {
        double dx = x[i] - meanX
        double dy = y[i] - meanY
        num += dx * dy
        denX += dx * dx
        denY += dy * dy
    }
    double den = Math.sqrt(denX * denY)
    return den > 0 ? num / den : Double.NaN
}

// Manders with thresholds
List<Double> calculateManders(double[] x, double[] y, double thX, double thY) {
    double sumX = 0, sumY = 0
    double sumXcoloc = 0, sumYcoloc = 0

    for (int i = 0; i < x.length; i++) {
        boolean xPos = x[i] > thX
        boolean yPos = y[i] > thY

        if (xPos) sumX += x[i]
        if (yPos) sumY += y[i]

        if (xPos && yPos) {
            sumXcoloc += x[i]
            sumYcoloc += y[i]
        }
    }

    double M1 = sumX > 0 ? sumXcoloc / sumX : Double.NaN
    double M2 = sumY > 0 ? sumYcoloc / sumY : Double.NaN
    return [M1, M2]
}

int ch1 = FIRST_CHANNEL - 1
int ch2 = SECOND_CHANNEL - 1

objectsToMeasure.eachWithIndex { obj, idx ->
    def roi = obj.getROI()
    if (roi == null) return

    try {
        // Request only ROI bounds (full-res)
        def request = RegionRequest.createInstance(server.getPath(), 1.0, roi)
        def img = server.readRegion(request)
        def raster = img.getRaster()

        int w = img.getWidth()
        int h = img.getHeight()
        int bands = raster.getNumBands()

        if (ch1 >= bands || ch2 >= bands) {
            print "Object ${obj.getID()}: channel index out of bounds (bands=${bands})"
            return
        }

        // ROI bounds in global coordinates
        int xMin = (int)Math.floor(roi.getBoundsX())
        int yMin = (int)Math.floor(roi.getBoundsY())
        int xMax = (int)Math.ceil(roi.getBoundsX() + roi.getBoundsWidth()) - 1
        int yMax = (int)Math.ceil(roi.getBoundsY() + roi.getBoundsHeight()) - 1

        def p1 = []
        def p2 = []

        // Convert global -> local image coords with request origin
        double rx = request.getX()
        double ry = request.getY()

        for (int y = yMin; y <= yMax; y++) {
            for (int x = xMin; x <= xMax; x++) {
                if (!roi.contains(x + 0.5, y + 0.5))
                    continue

                int lx = (int)Math.round(x - rx)
                int ly = (int)Math.round(y - ry)

                if (lx < 0 || ly < 0 || lx >= w || ly >= h)
                    continue

                p1 << raster.getSampleDouble(lx, ly, ch1)
                p2 << raster.getSampleDouble(lx, ly, ch2)
            }
        }

        if (p1.isEmpty()) return

        double[] a1 = p1 as double[]
        double[] a2 = p2 as double[]

        double pcc = calculatePCC(a1, a2)
        def manders = calculateManders(a1, a2, ch1Background, ch2Background)

        def ml = obj.getMeasurementList()
        ml.put("Colocalization: PCC_Ch${FIRST_CHANNEL}_Ch${SECOND_CHANNEL}", pcc)
        ml.put("Colocalization: M1_Ch${FIRST_CHANNEL}_over_Ch${SECOND_CHANNEL}", manders[0])
        ml.put("Colocalization: M2_Ch${SECOND_CHANNEL}_over_Ch${FIRST_CHANNEL}", manders[1])

        if (idx % 250 == 0)
            print "Processed ${idx + 1}/${objectsToMeasure.size()}"

    } catch (Exception e) {
        print "Object ${obj.getID()} error: ${e.getMessage()}"
    }
}

// Refresh hierarchy (version-safe)
try {
    hierarchy.fireObjectsChangedEvent(this, objectsToMeasure)
} catch (Exception e) {
    // fallback if your version exposes only global helper
    try {
        fireHierarchyUpdate()
    } catch (Exception ignored) {
        print "Could not fire hierarchy update automatically."
    }
}

print "Colocalization analysis complete."
