package code.common;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;

/**
 * Class Common
 * contains the attributes and the methods that are commonly used among other classes
 */
public class Common {

    public static HashMap<String, String> argumentsMap = new HashMap<>();
    public static List<String> fileNamesList = new ArrayList<>();

    public static final String sampleDirParamName = "sampleDirParam";
    public static final String appDirParamName = "appDirParam";
    public static final String outputDirParamName = "outputDirParam";
    public static final String annotationsPathFromParamName = "annotationsPathFromParam";
    public static final String annotationsPathToParamName = "annotationsPathToParam";
    public static final String extensionOutlinesFinalParamName = "extensionOutlinesFinalParam";
    public static final String cellSizeMinParamName = "cellSizeMinParam" ;
    public static final String cellSizeMaxParamName = "cellSizeMaxParam" ;

    /**
     * setParameter - replaces the parameter value in a string based on whether it's predefined or not
     * @param content - string in which the value should be replaced
     * @param paramName - value that shall be replaced
     * @param paramValue - new value to be set
     * @param isPredefined - states if the paramName has predefined value in the code or not
     * @return - String - updated string
     */
    public static String setParameter(String content, String paramName, String paramValue, boolean isPredefined) {

        if (isPredefined)
            return content.replace(paramName,
                    argumentsMap.get(paramName) != null ? argumentsMap.get(paramName) : paramValue);

        return content.replace(paramName, paramValue);
    }

    /**
     * setParameter - replaces the parameter value in a string with the value stored in the arguments map.
     * Used for paths from the config file
     * @param content - string in which the value should be replaced
     * @param paramName - value that shall be replaced
     * @return
     */
    public static String setParameter(String content, String paramName) {

        return content.replace(paramName,
                argumentsMap.get(paramName).replace("\\", "/"));
    }

    /**
     * getFileNameList - reads the names of the files present in the sampleDirectory
     * @param sampleDirectory - path to the directory
     * @param withExtension - states if the list should contain the file extensions or not
     * @return List<String> - list with the filenames
     */
    public static List<String> getFileNameList(File sampleDirectory, boolean withExtension) {
        List<String> fileNamesList = new ArrayList<>();

        // read filenames
        for (File file : Objects.requireNonNull(sampleDirectory.listFiles())) {

            if (file.isFile()) {
                String fileName = file.getName();

                if(!withExtension) {
                    int index = fileName.lastIndexOf('.');
                    fileName = fileName.substring(0, index); // only filename without extension
                }

                fileNamesList.add(fileName);
            }
        }
        return fileNamesList;
    }
}
