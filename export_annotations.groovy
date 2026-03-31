import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.objects.PathDetectionObject
import qupath.lib.regions.RegionRequest
import qupath.lib.awt.common.AwtTools

import java.awt.image.BufferedImage
import java.awt.Color
import javax.imageio.ImageIO

def imageData = getCurrentImageData()
def server = imageData.getServer()

// === OUTPUT ===
def name = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
def outDir = buildFilePath(PROJECT_BASE_DIR, "cellsighter_export", name)
mkdirs(outDir)

// === PARAMETERS ===
double requestedPixelSize = 1.0
int tileSize = 512
int overlap = 0

double downsample = requestedPixelSize / server.getPixelCalibration().getAveragedPixelSize()

// === SEGMENTATION MASK SERVER (ANNOTATIONS) ===
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, Color.BLACK)
    .downsample(downsample)
    .addLabel("Annotation", 1)   // ВСЕ аннотации → 1
    .multichannelOutput(false)   // бинарная маска
    .build()

// === ROI ===
def selected = getSelectedObject()
if (selected == null || selected.getROI() == null) {
    print "❌ Select ROI!"
    return
}

def roi = selected.getROI()
def bounds = AwtTools.getBounds(roi)

// === TILE LOOP ===
int tileIndex = 0

for (int y = bounds.y; y < bounds.y + bounds.height; y += tileSize) {
    for (int x = bounds.x; x < bounds.x + bounds.width; x += tileSize) {

        def region = RegionRequest.createInstance(
                server.getPath(), downsample,
                x, y, tileSize, tileSize
        )

        // === IMAGE TILE ===
        def img = server.readBufferedImage(region)
        if (img == null)
            continue

        // === SEGMENTATION MASK ===
        def mask = labelServer.readBufferedImage(region)

        // === POINT MASK ===
        BufferedImage pointMask = new BufferedImage(tileSize, tileSize, BufferedImage.TYPE_BYTE_GRAY)
        def g = pointMask.createGraphics()
        g.setColor(Color.BLACK)
        g.fillRect(0, 0, tileSize, tileSize)

        // === DRAW DETECTIONS ===
        def detections = getDetectionObjects()

        detections.each { det ->
            if (!(det instanceof PathDetectionObject))
                return

            def cls = det.getPathClass()
            if (cls == null)
                return

            def p = det.getROI().getCentroid()

            // проверяем попадает ли в тайл
            if (p.x >= x && p.x < x + tileSize &&
                p.y >= y && p.y < y + tileSize) {

                int px = (int)((p.x - x) / downsample)
                int py = (int)((p.y - y) / downsample)

                // === КЛАСС → INTENSITY ===
                int value = 255   // по умолчанию
                if (cls.getName() == "Positive")
                    value = 255
                else if (cls.getName() == "Negative")
                    value = 127

                pointMask.getRaster().setSample(px, py, 0, value)
            }
        }

        g.dispose()

        // === SAVE ===
        def base = String.format("tile_%05d", tileIndex)

        ImageIO.write(img, "PNG", new File(outDir, base + "_img.png"))
        ImageIO.write(mask, "PNG", new File(outDir, base + "_seg.png"))
        ImageIO.write(pointMask, "PNG", new File(outDir, base + "_points.png"))

        tileIndex++
    }
}

print "✅ Done! Tiles: " + tileIndex