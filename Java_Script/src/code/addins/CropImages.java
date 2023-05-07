package code.addins;

import code.common.Common;
import org.apache.commons.io.FileUtils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

public class CropImages {

    static final String cropBatchSizeParamName = "cropBatchSizeParam";
    static final String cropDirectoryParamName = "cropDirectoryParam";
    static final String cropTmpDirectoryParamName = "cropTmpDirectoryParam";
    static final String cropParamXParamName = "paramX";
    static final String cropParamYParamName = "paramY";
    static final String cropParamWidthParamName = "paramWidth";
    static final String cropParamHeightParamName = "paramHeight";
    static final String cropParamIsFirstRunParamName = "paramIsFirstRun";

    public Integer[] readCropSelectedOutlines(String fileNameOutline) {
        Integer[] cropSelectedOutlines = new Integer[4];
        String csvFile;
        String splitBy = "\n";
        String delimiter = ",";
        try {
            Scanner outlinesScanner = new Scanner(new File(fileNameOutline));
            csvFile = outlinesScanner.useDelimiter("\\Z").next();
            String[] lines = csvFile.split(splitBy);
            int a = 0;
            for (String line : lines) {
                a++;
                if (a == 1) continue;
                String[] data = line.split(delimiter);
                for (int i = 0; i <= cropSelectedOutlines.length - 1; i++) {
                    cropSelectedOutlines[i] = Integer.parseInt(data[i + 5]);
                }
            }
            outlinesScanner.close();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        return cropSelectedOutlines;
    }

    public void cropSamples() {

        boolean isCropSelectionNeeded = true;
        Integer[] cropSelection = null;
        Iterator<String> itr = Common.fileNamesList.listIterator();
        String cropMacroOutputPath = "src\\macros\\addins\\MacroImageCrop.ijm";
        String cropMacroTemplatePath = "src\\macros\\addins\\MacroImageCropTemplate.txt";
        String imageCropMacroCommand =  Common.argumentsMap.get(Common.appDirParamName) + " -macro " + System.getProperty("user.dir") + '\\' + cropMacroOutputPath;
        String tmpDirPath = Common.argumentsMap.get(Common.sampleDirParamName) + "\\tmp\\";
        String croppedDirPath = Common.argumentsMap.get(Common.outputDirParamName) + "\\cropped\\";
        String line;

        try {
            BufferedReader reader = new BufferedReader(new FileReader(cropMacroTemplatePath));
            File cropMacroFile = new File(cropMacroOutputPath);
            File tmpDirectory = new File(tmpDirPath);
            StringBuilder content = new StringBuilder();

            // add crop directory paths to code.common map
            Common.argumentsMap.put(cropTmpDirectoryParamName, tmpDirPath);
            Common.argumentsMap.put(cropDirectoryParamName, croppedDirPath);

            tmpDirectory.mkdir();

            // read macro template
            while ((line = reader.readLine()) != null) {
                content.append(line);
                content.append(System.lineSeparator());
            }
            reader.close();

            while (itr.hasNext()) {

                BufferedWriter writer = new BufferedWriter(new FileWriter(cropMacroFile));
                ProcessBuilder processBuilder = new ProcessBuilder(imageCropMacroCommand.split(" "));
                String updatedContent = content.toString();

                for (int i = 0; i < Integer.parseInt(Common.argumentsMap.get(cropBatchSizeParamName)) && itr.hasNext(); i++) {
                    String filename = itr.next();
                    Files.copy(
                            new File(Common.argumentsMap.get(Common.sampleDirParamName) + '\\' + filename).toPath(),
                            (new File(tmpDirPath + '\\' + filename)).toPath(),
                            StandardCopyOption.REPLACE_EXISTING);
                }

                updatedContent = Common.setParameter(updatedContent, cropTmpDirectoryParamName);
                updatedContent = Common.setParameter(updatedContent, cropDirectoryParamName);
                updatedContent = Common.setParameter(updatedContent, cropParamIsFirstRunParamName, String.valueOf(isCropSelectionNeeded), false);

                if (!isCropSelectionNeeded) {
                    updatedContent = Common.setParameter(updatedContent, cropParamXParamName, String.valueOf(cropSelection[0]), false);
                    updatedContent = Common.setParameter(updatedContent, cropParamYParamName, String.valueOf(cropSelection[1]), false);
                    updatedContent = Common.setParameter(updatedContent, cropParamWidthParamName, String.valueOf(cropSelection[2]), false);
                    updatedContent = Common.setParameter(updatedContent, cropParamHeightParamName, String.valueOf(cropSelection[3]), false);
                }

                writer.write(updatedContent);
                writer.close();

                // execute cropping macro
                Process process = processBuilder.start();
                process.waitFor();

                // crop selection is needed only for the first batch, the selection is then applied to the rest of the sample
                if (isCropSelectionNeeded) {
                    cropSelection = readCropSelectedOutlines(croppedDirPath + "\\region.csv");
                    isCropSelectionNeeded = false;
                }

                // clean directory
                FileUtils.cleanDirectory(tmpDirectory);
            }

            FileUtils.deleteDirectory(tmpDirectory);

            // work with cropped samples from now on
            Common.argumentsMap.replace(Common.sampleDirParamName, System.getProperty("user.dir") + "\\resources\\Outputs\\cropped\\");

        } catch (IOException | InterruptedException e) {
            System.out.println("An error occurred while creating or executing macro for cropping files : " + e.getMessage());
        }

        // delete the region selection file
        File regionFile = new File(croppedDirPath + "region.csv");
        if (!regionFile.delete()) {
            System.out.println("Failed to delete the region file.");
        }
    }
}
