package com.example;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class Evaluate {
    private static Logger log = LoggerFactory.getLogger(Evaluate.class);

    public static void main(String... args) throws Exception {
        Options options = new Options();

        options.addOption("i", "input", true, "The file with test data.");
        options.addOption("m", "model", true, "Name of trained model file.");

        CommandLine cmd = new BasicParser().parse(options, args);

        String input = cmd.getOptionValue("i");
        String modelName = cmd.getOptionValue("m");

        if (cmd.hasOption("i") && cmd.hasOption("m")) {
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelName);
            DataIterator<NormalizerStandardize> it = DataIterator.irisCsv(input);
            RecordReaderDataSetIterator testData = it.getIterator();
            NormalizerStandardize normalizer = it.getNormalizer();
            normalizer.load(
                    new File(modelName + ".norm1"),
                    new File(modelName + ".norm2"),
                    new File(modelName + ".norm3"),
                    new File(modelName + ".norm4")
            );

            Evaluation eval = new Evaluation(3);
            while (testData.hasNext()) {
                DataSet ds = testData.next();
                INDArray output = model.output(ds.getFeatureMatrix());
                eval.eval(ds.getLabels(), output);
            }

            log.info(eval.stats());
        } else {
            log.error("Invalid arguments.");

            new HelpFormatter().printHelp("Evaluate", options);
        }
    }
}
