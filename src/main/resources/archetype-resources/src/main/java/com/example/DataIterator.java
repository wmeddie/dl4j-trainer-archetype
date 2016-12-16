package com.example;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;


class DataIterator {
    private RecordReaderDataSetIterator iterator;
    private DataNormalization normalizer;

    private DataIterator(RecordReaderDataSetIterator it, DataNormalization norm) {
        this.iterator = it;
        this.normalizer = norm;
    }

    RecordReaderDataSetIterator getIterator() {
        return iterator;
    }

    DataNormalization getNormalizer() {
        return normalizer;
    }

    static DataIterator irisCsv(String name) {
        CSVRecordReader recordReader = new CSVRecordReader(0, ",");
        try {
            recordReader.initialize(new FileSplit(new File(name)));
        } catch (Exception e) {
            e.printStackTrace();
        }

        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 50;     //Iris data set: 150 examples total.

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(
                recordReader,
                batchSize,
                labelIndex,
                numClasses
        );

        NormalizerStandardize normalizer = new NormalizerStandardize();

        while (iterator.hasNext()) {
            normalizer.fit(iterator.next());
        }
        iterator.reset();

        iterator.setPreProcessor(normalizer);

        return new DataIterator(iterator, normalizer);
    }
}
