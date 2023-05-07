package code.common;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;

public class Common {

    public static final HashMap<String, String> argumentsMap = new HashMap<>();
    public static List<String> fileNamesList = new ArrayList<>();

    public static final String sampleDirParamName = "sampleDirParam";
    public static final String appDirParamName = "appDirParam";
    public static final String outputDirParamName = "outputDirParam";
    public static final String annotationsPathFromParamName = "annotationsPathFromParam";
    public static final String annotationsPathToParamName = "annotationsPathToParam";
    public static final String extensionOutlinesFinalParamName = "extensionOutlinesFinalParam";
    public static final String cellSizeMinParamName = "cellSizeMinParam" ;
    public static final String cellSizeMaxParamName = "cellSizeMaxParam" ;

    public static String setParameter(String content, String paramName, String paramValue, boolean isPredefined) {

        if (isPredefined)
            return content.replace(paramName,
                    argumentsMap.get(paramName) != null ? argumentsMap.get(paramName) : paramValue);

        return content.replace(paramName, paramValue);
    }

    public static String setParameter(String content, String paramName) {

        return content.replace(paramName,
                argumentsMap.get(paramName).replace("\\", "/"));
    }

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
