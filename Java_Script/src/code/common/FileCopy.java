package code.common;

import code.common.Common;
import org.apache.commons.io.FileUtils;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class FileCopy {

    public void runCopy(){

        File sampleDirectory = new File(Common.argumentsMap.get(Common.sampleDirParamName));
        List<String> fileNameList = Common.getFileNameList(sampleDirectory, false);

        for (String filename : fileNameList) {

            File source = new File(Common.argumentsMap.get(Common.annotationsPathFromParamName) + '\\' + filename+"."+
                    Common.argumentsMap.get(Common.extensionOutlinesFinalParamName));
            File dest = new File(Common.argumentsMap.get(Common.annotationsPathToParamName) + '\\' + filename+"." +
                    Common.argumentsMap.get(Common.extensionOutlinesFinalParamName));
            try {
                FileUtils.copyFile(source, dest);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}
