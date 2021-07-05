import org.apache.commons.csv.CSVFormat;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.measure.NominalScale;
import smile.data.vector.IntVector;
import smile.io.Read;
import smile.plot.swing.Histogram;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.URISyntaxException;
import java.util.Arrays;

public class TitanicSurvivorPredictor {
    public static void main(String[] args) throws Exception {
        DataFrame trainingData = readCSV("src/main/resources/titanic_train.csv");
        DataFrame testData = readCSV("src/main/resources/titanic_test.csv");

        trainingData = trainingData.select("Survived","Pclass","Sex","Age");

        trainingData = dataFrameEncoded(trainingData,"Pclass");
        trainingData = dataFrameEncoded(trainingData,"Sex");

        EDA(trainingData);

        testData = testData.select("Pclass","Sex","Age");
        testData = dataFrameEncoded(testData,"Sex");

        RandomForest model = RandomForest.fit(Formula.lhs("Survived"), trainingData);
        System.out.println("Features' importance:");
        System.out.println(Arrays.toString(model.importance()));
        System.out.println(model.metrics ());

    }

    public static DataFrame readCSV(String path) throws IOException, URISyntaxException {
        CSVFormat format = CSVFormat.DEFAULT.withFirstRecordAsHeader ();
        return Read.csv (path, format);
    }

    public static DataFrame dataFrameEncoded(DataFrame df, String columnName) {
        String[] values = df.stringVector (columnName).distinct ().toArray (new String[]{});
        int[] pclassValues = df.stringVector (columnName).factorize (new NominalScale(values)).toIntArray ();
        df = df.merge (IntVector.of (columnName+" Encoded", pclassValues));
        df = df.drop(columnName);
        return df;
    }

    public static void EDA(DataFrame df) throws InterruptedException, InvocationTargetException {
        DataFrame titanicSurvived = DataFrame.of (df.stream ().filter (t -> t.get ("Survived").equals (1)));
        DataFrame titanicNotSurvived = DataFrame.of (df.stream ().filter (t -> t.get ("Survived").equals (0)));

        titanicSurvived = titanicSurvived.omitNullRows ();
        titanicNotSurvived = titanicNotSurvived.omitNullRows ();

        System.out.println(titanicSurvived.summary());
        System.out.println(titanicNotSurvived.summary());

        System.out.println (titanicSurvived.schema ());
        System.out.println (titanicNotSurvived.schema ());

        Histogram.of (titanicSurvived.doubleVector ("Age").toDoubleArray (), 15, false)
                .canvas ().setAxisLabels ("Age", "Count")
                .setTitle ("Age frequencies among surviving passengers")
                .window ();
        Histogram.of (titanicSurvived.intVector ("Pclass Encoded").toIntArray (), 4, true)
                .canvas ().setAxisLabels ("Classes", "Count")
                .setTitle ("Pclass values frequencies among surviving passengers")
                .window ();
        Histogram.of (titanicNotSurvived.doubleVector ("Age").toDoubleArray (), 15, false)
                .canvas ().setAxisLabels ("Age", "Count")
                .setTitle ("Age frequencies among not surviving passengers")
                .window ();
        Histogram.of (titanicNotSurvived.intVector ("Pclass Encoded").toIntArray (), 4, true)
                .canvas ().setAxisLabels ("Classes", "Count")
                .setTitle ("Pclass values frequencies among not surviving passengers")
                .window ();
    }
}
