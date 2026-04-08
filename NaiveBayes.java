import java.util.*;

public class NaiveBayes {

    static Map<String, Integer> spamWords = new HashMap<>();
    static Map<String, Integer> hamWords = new HashMap<>();
    static int spamCount = 0, hamCount = 0;
    static int totalSpamWords = 0, totalHamWords = 0;
    static Set<String> vocabulary = new HashSet<>();

    // Training function
    static void train(String text, String label) {
        String[] words = text.split(" ");

        if (label.equals("Spam")) {
            spamCount++;
            for (String w : words) {
                vocabulary.add(w);
                spamWords.put(w, spamWords.getOrDefault(w, 0) + 1);
                totalSpamWords++;
            }
        } else {
            hamCount++;
            for (String w : words) {
                vocabulary.add(w);
                hamWords.put(w, hamWords.getOrDefault(w, 0) + 1);
                totalHamWords++;
            }
        }
    }

    // Prediction function
    static String predict(String text) {
        String[] words = text.split(" ");

        double spamProb = Math.log((double) spamCount / (spamCount + hamCount));
        double hamProb = Math.log((double) hamCount / (spamCount + hamCount));

        for (String w : words) {
            double wordSpam = (spamWords.getOrDefault(w, 0) + 1.0) /
                    (totalSpamWords + vocabulary.size());

            double wordHam = (hamWords.getOrDefault(w, 0) + 1.0) /
                    (totalHamWords + vocabulary.size());

            spamProb += Math.log(wordSpam);
            hamProb += Math.log(wordHam);
        }

        return spamProb > hamProb ? "Spam" : "Ham";
    }

    public static void main(String[] args) {

        // Training Data
        train("win money now", "Spam");
        train("cheap money offer", "Spam");
        train("meeting schedule today", "Ham");
        train("project discussion meeting", "Ham");

        // Test Data
        String[] testDocs = {
                "win money",
                "schedule meeting",
                "cheap offer"
        };

        String[] actual = {"Spam", "Ham", "Spam"};

        int TP = 0, TN = 0, FP = 0, FN = 0;

        for (int i = 0; i < testDocs.length; i++) {
            String predicted = predict(testDocs[i]);
            System.out.println("Text: " + testDocs[i]);
            System.out.println("Predicted: " + predicted + " | Actual: " + actual[i]);
            System.out.println();

            if (predicted.equals("Spam") && actual[i].equals("Spam")) TP++;
            else if (predicted.equals("Ham") && actual[i].equals("Ham")) TN++;
            else if (predicted.equals("Spam") && actual[i].equals("Ham")) FP++;
            else if (predicted.equals("Ham") && actual[i].equals("Spam")) FN++;
        }

        double accuracy = (double)(TP + TN) / (TP + TN + FP + FN);
        double precision = (double) TP / (TP + FP);
        double recall = (double) TP / (TP + FN);

        System.out.println("Accuracy = " + accuracy);
        System.out.println("Precision = " + precision);
        System.out.println("Recall = " + recall);
    }
}