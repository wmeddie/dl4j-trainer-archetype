# $name$

An awesome Scala-based DeepLearning4J project.

## Training

Use sbt to train the model.

    sbt "run-main $organization$.$name;format="lower,word"$.Train --input trainInput --output output.model --epoch 5"

Example:
    sbt "run-main $organization$.$name;format="lower,word"$.Train --input iris.train.csv --output out1.model --epoch 10"

## Evaluation

    sbt "run-main $organization$.$name;format="lower,word"$.Evaluate --input testInput --model trained.model"

Example:
    sbt "run-main $organization$.$name;format="lower,word"$.Evaluate --input iris.test.csv --model out1.model"

## Repl

Play around with your model with the repl.  It'll import most of DL4J's important classes by default.

    sbt console

