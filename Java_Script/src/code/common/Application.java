package code.common;

import code.addins.ComputeCellArea;
import code.addins.CropImages;
import code.evaluation.Stats;
import code.processing.ProcessImages;

import java.io.File;
import java.util.Objects;
/**
 *  Class Application
 *  runs the app
 */
public class Application {

    private final String cropSampleParamName = "cropSample" ;
    private final String computeCellSizeParamName = "computeCellSize";
    private final String yesParamValue = "y" ;
    private final String noParamValue = "n" ;
    private final String datasetNameParamName = "datasetNameParam" ;
    private final String runProcessingParamName = "runProcessingParam" ;

    /**
     * run - runs the application according to inputs from the config file
     */
    public void run() {

        ReadConfig reader = new ReadConfig();

        reader.readConfigFile();

        File sampleDirectory = new File(Common.argumentsMap.get(Common.sampleDirParamName));
        Common.fileNamesList = Common.getFileNameList(sampleDirectory, true);

        // if requested compute cell area in pixels
        if (Objects.equals(Common.argumentsMap.get(computeCellSizeParamName), yesParamValue))
        {
            ComputeCellArea computer = new ComputeCellArea();
            computer.computeCellArea();
        }

        // if requested crop samples
        if (Objects.equals(Common.argumentsMap.get(cropSampleParamName), yesParamValue))
        {
            CropImages cropper = new CropImages();
            cropper.cropSamples();
        }

        // if requested to process samples
        if (Objects.equals(Common.argumentsMap.get(runProcessingParamName), yesParamValue))
        {
            ProcessImages processor = new ProcessImages();
            processor.processImages();
        }

        // statistics - only on not cropped images
        if (Objects.equals(Common.argumentsMap.get(cropSampleParamName), noParamValue)) {

            Stats stats = new Stats();

            // copy JSON annotations
            FileCopy fileCopy = new FileCopy();
            fileCopy.runCopy();

            stats.run(Common.argumentsMap.get(datasetNameParamName) != null ?
                    Common.argumentsMap.get(datasetNameParamName) : "Dataset");
        }
    }
}
