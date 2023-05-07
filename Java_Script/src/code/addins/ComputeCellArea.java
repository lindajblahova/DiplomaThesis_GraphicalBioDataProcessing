package code.addins;

import code.common.Common;

public class ComputeCellArea {

    final String zoomParamName = "zoomParam" ;
    final String cellMinDiameterParamName = "cellMinDiameterParam" ;
    final String cellMaxDiameterParamName = "cellMaxDiameterParam" ;
    final String pixelSizeParamName = "pixelSizeParam" ;
    final String fovParamName = "fovParam" ;

    private double[] calculation(int zoom, int cellMinDiameter, int cellMaxDiameter, double pixelSize, double fieldOfView) {

        double areaMinMicrometers = Math.PI * Math.pow((double) cellMinDiameter/2, 2);
        double areaMaxMicrometers = Math.PI * Math.pow((double) cellMaxDiameter/2, 2);

        double areaMinPixels = (areaMinMicrometers * zoom) / (Math.pow(pixelSize, 2) * fieldOfView);
        double areaMaxPixels = (areaMaxMicrometers * zoom) / (Math.pow(pixelSize, 2) * fieldOfView);

        return new double[]{areaMinPixels,areaMaxPixels};
    }

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
