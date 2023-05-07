package code.processing;

import code.common.Common;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Class ProcessImages
 * process the images using ImageJ
 */
public class ProcessImages {

    private HashMap<String,String> predefinedValuesMap = new HashMap<>();

    private final String statsOutputDirParamName = "statsOutputDirParam";
    private final String fileNameParamName = "fileNameParam";
    private final String enhanceContrastParamName = "enhContrastParam" ;
    private final String dilateCountParamName = "dilateCountParam";
    private final String isLightStdParamName = "isLightSTDParam";
    private final String isNotLightParamName = "isNotLightParam";
    private final String isNeutralParamName = "isNeutralParam";
    private final String isNeutralStdParamName = "isNeutralSTDParam";
    private  final String isDarkStdParamName = "isDarkSTDParam";
    private final String isNotDarkParamName = "isNotDarkParam";
    private final String lightMinParamName = "lightMinParam";
    private final String lightMaxParamName = "lightMaxParam";
    private final String neutralMinParamName = "neutralMinParam";
    private final String neutralMaxParamName = "neutralMaxParam";
    private final String darkMinParamName = "darkMinParam";
    private final String darkMaxParamName = "darkMaxParam";

    /**
     * Constructor
     */
    public ProcessImages() {
        createPredefinedValuesMap();
    }

    /**
     * Constructor - fills the map with the arguments that have predefined values
     */
    public void createPredefinedValuesMap() {
        this.predefinedValuesMap.put(enhanceContrastParamName, "0.40");
        this.predefinedValuesMap.put(dilateCountParamName, "4");
        this.predefinedValuesMap.put(isLightStdParamName, "4");
        this.predefinedValuesMap.put(isNotLightParamName, "4");
        this.predefinedValuesMap.put(isNeutralParamName, "4");
        this.predefinedValuesMap.put(isNeutralStdParamName, "4");
        this.predefinedValuesMap.put(isDarkStdParamName, "4");
        this.predefinedValuesMap.put(isNotDarkParamName, "4");
        this.predefinedValuesMap.put(lightMinParamName, "4");
        this.predefinedValuesMap.put(lightMaxParamName, "4");
        this.predefinedValuesMap.put(neutralMinParamName, "4");
        this.predefinedValuesMap.put(neutralMaxParamName, "4");
        this.predefinedValuesMap.put(darkMinParamName, "4");
        this.predefinedValuesMap.put(darkMaxParamName, "4");
        this.predefinedValuesMap.put(Common.cellSizeMinParamName, "8000");
        this.predefinedValuesMap.put(Common.cellSizeMaxParamName, "103000");
    }

    /**
     * processImages - method that runs the image processing using ImageJ macros and updates the macros correspondingly
     */
    public void processImages() {
        String macroTemplatePath = "src\\macros\\processing\\MacroImageProcessingTemplate.txt";
        String macroOutputPath = "src\\macros\\processing\\MacroImageProcessing.ijm";
        Common.argumentsMap.put(statsOutputDirParamName, Common.argumentsMap.get(Common.outputDirParamName) + "\\stats\\");
        String imageProcessMacroCommand = Common.argumentsMap.get(Common.appDirParamName) + " -macro " + System.getProperty("user.dir") + '\\' + macroOutputPath;

        try {
            File macroTemplateFile = new File(macroTemplatePath);
            File macroImageProcessingFile = new File(macroOutputPath);
            BufferedReader reader = new BufferedReader(new FileReader(macroTemplateFile));
            StringBuilder content = new StringBuilder();
            String line;

            // read macro template
            while ((line = reader.readLine()) != null) {
                content.append(line);
                content.append(System.lineSeparator());
            }
            reader.close();

            for (String name : Common.fileNamesList) {

                // update  macro template
                String[] name_array = name.split("\\.",2);
                String updatedContent = content.toString();
                updatedContent = Common.setParameter(updatedContent, fileNameParamName, name_array[0] , false);
                updatedContent = Common.setParameter(updatedContent, Common.sampleDirParamName);
                updatedContent = Common.setParameter(updatedContent, statsOutputDirParamName);

                for(Map.Entry<String, String> value : predefinedValuesMap.entrySet()) {
                    updatedContent = Common.setParameter(updatedContent, value.getKey(), value.getValue(), true);
                }

                // write image processing macro to execute
                BufferedWriter writer = new BufferedWriter(new FileWriter(macroImageProcessingFile));
                writer.write(updatedContent);
                writer.close();

                // execute image processing macro
                ProcessBuilder processBuilder = new ProcessBuilder(imageProcessMacroCommand.split(" "));
                Process process = processBuilder.start();
                process.waitFor();
            }

        } catch (IOException | InterruptedException e) {
            System.out.println("An error occurred while creating or executing macro : " + e.getMessage());
        }
    }
}
