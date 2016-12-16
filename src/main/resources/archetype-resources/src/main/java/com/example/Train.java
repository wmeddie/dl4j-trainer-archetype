package com.example;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Train {
    private static Logger log = LoggerFactory.getLogger(Train.class);

    private static MultiLayerConfiguration net(int nIn, int nOut) {
        return new NeuralNetConfiguration.Builder()
                .seed(42)
                .iterations(1)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1)
                .regularization(true).l2(1e-4)
                .list(
                        new DenseLayer.Builder().nIn(nIn).nOut(3).build(),
                        new DenseLayer.Builder().nIn(3).nOut(3).build(),
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation("softmax")
                                .nIn(3)
                                .nOut(nOut)
                                .build()
                )
                .build();
    }

    public static void main(String... args) throws Exception {
        Options options = new Options();

        options.addOption("i", "input", true, "The file with training data.");
        options.addOption("o", "output", true, "Name of trained model file.");
        options.addOption("e", "epoch", true, "Number of times to go over whole training set.");

        CommandLine cmd = new BasicParser().parse(options, args);

        if (cmd.hasOption("i") && cmd.hasOption("o") && cmd.hasOption("e")) {
            train(cmd);
            log.info("Training finished.");
        } else {
            log.error("Invalid arguments.");

            new HelpFormatter().printHelp("Train", options);
        }
    }

    private static void train(CommandLine c) {
        int nEpochs = Integer.parseInt(c.getOptionValue("e"));
        String modelName = c.getOptionValue("o");
        DataIterator it = DataIterator.irisCsv(c.getOptionValue("i"));
        RecordReaderDataSetIterator trainData = it.getIterator();
        DataNormalization normalizer = it.getNormalizer();

        log.info("Data Loaded");

        MultiLayerConfiguration conf = net(4, 3);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);

        model.init();

        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(Arrays.asList(new ScoreIterationListener(1), new StatsListener(statsStorage)));

        for (int i = 0; i < nEpochs; i++) {
            log.info("Starting epoch {} of {}", i, nEpochs);

            while (trainData.hasNext()) {
                model.fit(trainData.next());
            }

            log.info("Finished epoch {}", i);
            trainData.reset();
        }

        try {
            ModelSerializer.writeModel(model, modelName, true);

            normalizer.save(
                    new File(modelName + ".norm1"),
                    new File(modelName + ".norm2"),
                    new File(modelName + ".norm3"),
                    new File(modelName + ".norm4")
            );
        } catch (IOException e) {
            e.printStackTrace();
        }

        log.info("Model saved to: {}", modelName);
    }
}
