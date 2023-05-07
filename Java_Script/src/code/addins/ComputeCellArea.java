package code.addins;

import code.common.Common;

/**
 *  Class ComputeCellArea
 *  calculates estimated cell area in pixels
 */
public class ComputeCellArea {

    private final String zoomParamName = "zoomParam" ;
    private final String cellMinDiameterParamName = "cellMinDiameterParam" ;
    private final String cellMaxDiameterParamName = "cellMaxDiameterParam" ;
    private final String pixelSizeParamName = "pixelSizeParam" ;
    private final String fovParamName = "fovParam" ;

    /**
     *  calculation - performs calculation of estimated cell area based on arguments
     * @param zoom - used microscope zoom
     * @param cellMinDiameter - minimum cell diameter in micrometers
     * @param cellMaxDiameter - maximum cell diameter in micrometers
     * @param pixelSize - pixel size of the camera used in micrometers
     * @param fieldOfView - field of view of the microscope used
     * @return double[] - estimated interval of the cell area
     */
    private double[] calculation(int zoom, int cellMinDiameter, int cellMaxDiameter, double pixelSize, double fieldOfView) {

        double areaMinMicrometers = Math.PI * Math.pow((double) cellMinDiameter/2, 2);
        double areaMaxMicrometers = Math.PI * Math.pow((double) cellMaxDiameter/2, 2);

        double areaMinPixels = (areaMinMicrometers * zoom) / (Math.pow(pixelSize, 2) * fieldOfView);
        double areaMaxPixels = (areaMaxMicrometers * zoom) / (Math.pow(pixelSize, 2) * fieldOfView);

        return new double[]{areaMinPixels,areaMaxPixels};
    }

    /**
     *  computeCellArea - gets the necessary arguments and stores the results in the arguments map
     */
    public void computeCellArea() {
        double[] computedCellAreaInterval = calculation(Integer.parseInt(Common.argumentsMap.get(zoomParamName)),
                Integer.parseInt(Common.argumentsMap.get(cellMinDiameterParamName)),
                Integer.parseInt(Common.argumentsMap.get(cellMaxDiameterParamName)),
                Double.parseDouble(Common.argumentsMap.get(pixelSizeParamName)),
                Double.parseDouble(Common.argumentsMap.get(fovParamName)));

        Common.argumentsMap.put(Common.cellSizeMinParamName, String.valueOf(computedCellAreaInterval[0]));
        Common.argumentsMap.put(Common.cellSizeMaxParamName, String.valueOf(computedCellAreaInterval[1]));
    }
}
