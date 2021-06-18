package chapter2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/*分類器のテストを行うプログラムその２　k分割交差確認を行う*/

public class TestClassify2 {
	public static void main(String[] args) {
		int k = 10;	//交差確認法における分割数

		LearningDataSet dataSet = LearningDataSet.readFromFile("person_data.txt");
		List<LabeledVector> lvList = dataSet.labeledVectorList;
		Collections.shuffle(lvList);	//学習データの順番をランダムに並び替える

		int[][] confusionMatrix = new int [dataSet.maxLabel + 1][dataSet.maxFeature + 1];
		int numCorrect = 0;	//正解数

		//学習データとテストデータの組み合わせを変えてk回繰り返す
		for(int i = 0; i < k; i++) {
			//学習データとテストデータの振り分け
			List<LabeledVector> trainingDataList = new ArrayList<LabeledVector>();
			List<LabeledVector> testDataList = new ArrayList<LabeledVector>();
			for(int j = 0; j < lvList.size(); j++) {
				if(j % k == i) {
					testDataList.add(lvList.get(j));
				}else {
					trainingDataList.add(lvList.get(j));
				}
			}

			//分類器の初期化
			//Classifier classifier = new NaiveBayesClassifier();
			//Classifier classifier = new SVMClassifier();
			Classifier classifier = new NeuralClassifier();

			//学習データを用いた学習
			classifier.train(trainingDataList, dataSet.maxLabel, dataSet.maxFeature);

			//テストデータへの適用
			for(LabeledVector lv : testDataList) {
				int c = classifier.classify(lv.featureVector);
				System.out.println("名前 = " + lv.name + "\t正解ラベル = " + lv.label + "\t分類結果 = " + c);
				confusionMatrix[c][lv.label]++;
				if(c == lv.label) {
					numCorrect++;
				}
			}
		}
		System.out.println("混同行列");
		for(int i = 1; i <= dataSet.maxLabel; i++) {
			for(int j = 1; j <= dataSet.maxLabel; j++) {
				System.out.print(confusionMatrix[i][j] + "\t");
			}
			System.out.println();
		}
		System.out.println("正解率 = " + ((double)numCorrect / lvList.size()));
	}
}
