/* QuPath-Script for exporting square annotations (images and masks) with cell DETECTIONS
Exporting RGB, multichannel and single-channel ROIs

Need to add a case if annotations instead of cell detections (create cell detections without cytoplasm)
*/

import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RoiTools
import qupath.lib.common.GeneralTools
import ij.ImagePlus
import ij.process.ShortProcessor
import ij.IJ
import ij.process.ImageProcessor

import java.awt.Color
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.Shape
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

// =====================================
// USER SETTINGS
// =====================================

double downsample = 1.0
String exportRootName = "ground_truth"
String imageExt = ".tif"
String maskSuffix = "_masks"

boolean exportSelectedAnnotationsOnly = false
boolean skipUnclassifiedCells = false
Integer channelOfInterest = null  // null - все каналы, 1,2,3... - конкретный канал

// =====================================
// HELPERS
// =====================================

def csvEscape = { String s ->
    if (s == null) return ""
    String t = s.replace("\"", "\"\"")
    if (t.contains(",") || t.contains("\"") || t.contains("\n") || t.contains("\r"))
        return "\"${t}\""
    return t
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

def getChannelNames = { server ->
    def out = []
    try {
        def metadata = server.getMetadata()
        def channels = metadata?.getChannels()
        if (channels != null && !channels.isEmpty()) {
            channels.eachWithIndex { ch, i ->
                String nm = ch?.getName()
                if (nm == null || nm.isBlank()) nm = "Channel_${i + 1}"
                out << nm
            }
        }
    } catch (Exception e) { }
    if (out.isEmpty()) {
        try {
            int n = server.nChannels()
            for (int i = 0; i < n; i++) out << "Channel_${i + 1}"
        } catch (Exception e) { out << "Channel_1" }
    }
    return out
}

// Функция для извлечения одного канала из BufferedImage
def extractSingleChannel(BufferedImage img, int channelNumber, int totalChannels) {
    if (totalChannels <= 1) {
        println "  Изображение одноканальное, возвращаем как есть"
        return img
    }
    
    if (channelNumber > totalChannels) {
        println "  ВНИМАНИЕ: Канал ${channelNumber} не существует (всего ${totalChannels} каналов)"
        println "  Возвращаем все каналы"
        return img
    }
    
    try {
        // Конвертируем BufferedImage в ImagePlus
        def imp = new ImagePlus("Temp", img)
        def processor = imp.getProcessor()
        
        // Создаем новый ImagePlus для одного канала
        def channelImp = new ImagePlus("Channel", processor)
        
        // Извлекаем нужный канал (для RGB: 0=R,1=G,2=B)
        int channelIndex = channelNumber - 1
        if (channelIndex >= 0 && channelIndex < totalChannels) {
            // Для RGB изображений
            if (totalChannels == 3 && img.getType() == BufferedImage.TYPE_INT_RGB) {
                def channelProcessor = new ij.process.ColorProcessor(img)
                def channelImage = channelProcessor.getChannel(channelIndex + 1)
                return channelImage.getBufferedImage()
            } else {
                // Для других мультиканальных изображений
                def channels = ij.plugin.ChannelSplitter.split(imp)
                if (channelNumber <= channels.length) {
                    def result = channels[channelNumber - 1].getBufferedImage()
                    channels.each { if (it != null) it.close() }
                    return result
                }
            }
        }
        imp.close()
    } catch (Exception e) {
        println "  Ошибка при извлечении канала: ${e.getMessage()}"
    }
    return img
}

// =====================================
// INITIAL SETUP
// =====================================

def imageData = getCurrentImageData()
def server = imageData.getServer()
def imageName = GeneralTools.stripExtension(server.getMetadata().getName())

def cal = server.getPixelCalibration()
double pixelWidth = cal.getPixelWidth()
double pixelHeight = cal.getPixelHeight()
String pixelUnit = cal.getPixelWidthUnit()
println "Original calibration: ${pixelWidth} x ${pixelHeight} ${pixelUnit}/pixel"

// Получаем количество каналов
int nChannels = server.nChannels()
println "Total channels in image: ${nChannels}"

if (channelOfInterest != null) {
    println "Channel of interest: ${channelOfInterest}"
    if (channelOfInterest > nChannels) {
        println "WARNING: Channel ${channelOfInterest} does not exist! Will export all channels."
    } else if (nChannels == 1) {
        println "NOTE: Image is single-channel, channel_of_interest setting will be ignored"
    }
} else {
    println "Exporting ALL channels"
}

if (getProject() == null) {
    print "ERROR: Please open a QuPath project first."
    return
}

def channelNames = getChannelNames(server)
println "Channel names: ${channelNames}"

def exportRoot = buildFilePath(PROJECT_BASE_DIR, exportRootName)
mkdirs(exportRoot)

def channelsPath = buildFilePath(exportRoot, "channels.txt")
writeTextFile(channelsPath, channelNames.join("\n") + "\n")

def exportSummaryPath = buildFilePath(exportRoot, "export_summary.csv")
writeTextFile(exportSummaryPath, "index,image_name,mask_name,source_square,cell_count\n")

// =====================================
// GET SQUARE ANNOTATIONS AND DETECTIONS
// =====================================

def squareAnnotations = getAnnotationObjects().findAll { 
    it.getROI() instanceof qupath.lib.roi.RectangleROI 
}

if (squareAnnotations.isEmpty()) {
    print "ERROR: No square annotations found."
    return
}

print "Found ${squareAnnotations.size()} square annotation(s) to export."

def allDetections = getDetectionObjects()
println "Total detections in image: ${allDetections.size()}"

// =====================================
// EXPORT EACH SQUARE
// =====================================

int exportedCount = 0

squareAnnotations.eachWithIndex { square, idx ->
    
    def squareROI = square.getROI()
    String squareName = square.getName()
    if (squareName == null || squareName.isBlank()) squareName = "square_${idx + 1}"
    
    int x0 = (int)Math.floor(squareROI.getBoundsX())
    int y0 = (int)Math.floor(squareROI.getBoundsY())
    int w = (int)Math.ceil(squareROI.getBoundsWidth())
    int h = (int)Math.ceil(squareROI.getBoundsHeight())
    
    def cellsInSquare = allDetections.findAll { detection ->
        def roi = detection.getROI()
        if (roi == null) return false
        double cx = roi.getCentroidX()
        double cy = roi.getCentroidY()
        return (cx >= x0 && cx <= x0 + w && cy >= y0 && cy <= y0 + h)
    }
    
    if (skipUnclassifiedCells) {
        cellsInSquare = cellsInSquare.findAll { it.getPathClass() != null }
    }
    
    println ""
    println "====================================="
    println "Exporting square ${idx + 1}/${squareAnnotations.size()}: ${squareName}"
    println "Cells kept: ${cellsInSquare.size()}"
    println "Region (pixels): x=${x0}, y=${y0}, w=${w}, h=${h}"
    
    int imgNumber = idx + 1
    String imageFileName = "img_${imgNumber}${imageExt}"
    String maskFileName = "img_${imgNumber}${maskSuffix}${imageExt}"
    
    def imagePath = buildFilePath(exportRoot, imageFileName)
    def maskPath = buildFilePath(exportRoot, maskFileName)
    
    def request = RegionRequest.createInstance(server.getPath(), downsample, x0, y0, w, h)
    
    BufferedImage img = null
    try {
        img = server.readRegion(request)
    } catch (Exception e) {
        println "ERROR: failed to read image: ${e.getMessage()}"
        return
    }
    
    if (img == null) {
        println "ERROR: server.readRegion returned null"
        return
    }
    
    int outW = img.getWidth()
    int outH = img.getHeight()
    println "Exported size: ${outW} x ${outH}"
    
    // ========== ОБРАБОТКА КАНАЛОВ ==========
    BufferedImage imgToSave = img
    
    // Определяем, нужно ли извлекать один канал
    boolean extractChannel = (channelOfInterest != null && 
                              nChannels > 1 && 
                              channelOfInterest <= nChannels)
    
    if (extractChannel) {
        println "  Extracting channel ${channelOfInterest}..."
        imgToSave = extractSingleChannel(img, channelOfInterest, nChannels)
        println "  Channel extracted, new size: ${imgToSave.getWidth()} x ${imgToSave.getHeight()}"
    } else if (channelOfInterest != null && nChannels == 1) {
        println "  Image is single-channel, saving as is"
    } else if (channelOfInterest != null && channelOfInterest > nChannels) {
        println "  Channel ${channelOfInterest} not available, saving all channels"
    } else {
        println "  Saving all channels"
    }
    
    // Сохраняем изображение
    try {
        // Для TIFF используем IJ.save (сохраняет калибровку)
        def impToSave = new ImagePlus(imageFileName, imgToSave)
        def calToSave = impToSave.getCalibration()
        calToSave.setUnit(pixelUnit)
        calToSave.pixelWidth = pixelWidth
        calToSave.pixelHeight = pixelHeight
        IJ.save(impToSave, imagePath)
        impToSave.close()
        println "  Image saved: ${imageFileName}"
    } catch (Exception e) {
        println "ERROR: failed to write image: ${e.getMessage()}"
        return
    }
    
    // ========== СОЗДАНИЕ МАСКИ (НЕ ЗАВИСИТ ОТ КАНАЛОВ) ==========
    short[] pixels = new short[outW * outH]
    
    double scaleX = outW / (double)w
    double scaleY = outH / (double)h
    
    int cellId = 1
    
    cellsInSquare.eachWithIndex { cell, cellIdx ->
        
        def roi = cell.getROI()
        
        AffineTransform tx = new AffineTransform()
        tx.translate(-x0, -y0)
        tx.scale(scaleX, scaleY)
        
        Shape shape = RoiTools.getShape(roi)
        Shape transformedShape = tx.createTransformedShape(shape)
        
        BufferedImage tmp = new BufferedImage(outW, outH, BufferedImage.TYPE_BYTE_GRAY)
        Graphics2D g2d = tmp.createGraphics()
        g2d.setBackground(Color.BLACK)
        g2d.clearRect(0, 0, outW, outH)
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        g2d.setColor(Color.WHITE)
        g2d.fill(transformedShape)
        g2d.dispose()
        
        def tmpRaster = tmp.getRaster()
        def bounds = transformedShape.getBounds()
        
        int bx0 = Math.max(0, bounds.x)
        int by0 = Math.max(0, bounds.y)
        int bx1 = Math.min(outW, bounds.x + bounds.width)
        int by1 = Math.min(outH, bounds.y + bounds.height)
        
        for (int yy = by0; yy < by1; yy++) {
            for (int xx = bx0; xx < bx1; xx++) {
                if (tmpRaster.getSample(xx, yy, 0) > 0) {
                    pixels[yy * outW + xx] = (short)cellId
                }
            }
        }
        
        cellId++
        
        if ((cellIdx + 1) % 50 == 0) {
            println "  Rasterized ${cellIdx + 1}/${cellsInSquare.size()} cells"
        }
    }
    
    // Создаем ImagePlus для маски
    ShortProcessor sp = new ShortProcessor(outW, outH, pixels, null)
    ImagePlus maskImp = new ImagePlus("Mask", sp)
    
    def maskCal = maskImp.getCalibration()
    maskCal.setUnit(pixelUnit)
    maskCal.pixelWidth = pixelWidth
    maskCal.pixelHeight = pixelHeight
    
    IJ.save(maskImp, maskPath)
    maskImp.close()
    
    println "Saved:"
    println "  image -> ${imageFileName} (${outW}x${outH}, ${pixelWidth} ${pixelUnit}/px)"
    println "  mask  -> ${maskFileName} (${outW}x${outH}, ${maskCal.pixelWidth} ${maskCal.getUnit()}/px)"
    println "  ✓ Calibration matches!"
    
    appendTextFile(exportSummaryPath, 
        "${imgNumber},${imageFileName},${maskFileName},${csvEscape(squareName)},${cellsInSquare.size()}\n")
    
    exportedCount++
}

println ""
println "====================================="
println "ALL DONE."
println "Export folder: ${exportRoot}"
println "Images exported: ${exportedCount}"
println "Channels file: ${channelsPath}"
println "Export summary: ${exportSummaryPath}"
