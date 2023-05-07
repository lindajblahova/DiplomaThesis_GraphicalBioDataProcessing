package code.common;

import org.apache.commons.io.FileUtils;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Class FileCopy
 * copies the dataset annotations
 */
public class FileCopy {

    private String extensionOutlinesFinalParamValue = "json";

    /**
     * runCopy - copies the annotations from the desired path to the desired path, where both are set in the config file
     */
    public void runCopy(){

        File sampleDirectory = new File(Common.argumentsMap.get(Common.sampleDirParamName));
        List<String> fileNameList = Common.getFileNameList(sampleDirectory, false);

        String extension = Common.argumentsMap.get(Common.extensionOutlinesFinalParamName) != null ?
                Common.argumentsMap.get(Common.extensionOutlinesFinalParamName) : extensionOutlinesFinalParamValue;

        for (String filename : fileNameList) {

            File source = new File(Common.argumentsMap.get(Common.annotationsPathFromParamName) + '\\' + filename+"."+
                    extension);
            File dest = new File(Common.argumentsMap.get(Common.annotationsPathToParamName) + '\\' + filename+"." +
                    extension);
            try {
                FileUtils.copyFile(source, dest);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}
