package chapter2;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/*ナイーブベイズ分類器*/

public class NaiveBayesClassifier implements Classifier {
	/*スムージングのパラメータ*/
	public double alpha = 1.0;
	/*ラベルの最大値*/
	public int  maxLabel;
	/*特徴のインデックスの最大値*/
	public int maxFeature;
	/*各ラベルの出現回数*/
	public int[] labelCount;
	/*特徴ベクトルの各要素の出現回数*/
	public int[][] featureCount;
	/*各ラベルの出現確率*/
	public double[] labelProbability;
	/*特徴ベクトルの各要素の出現確率*/
	public double[][] featureProbability;

	/*学習データセットを使って学習を行う*/
	@Override
	public void train(List<LabeledVector> trainingDataSet, int maxLabel, int maxFeature) {
		this.maxLabel = maxLabel;
		this.maxFeature = maxFeature;

		labelCount = new int[maxLabel + 1];
		featureCount = new int[maxLabel + 1][maxFeature + 1];
		labelProbability = new double[maxLabel + 1];
		featureProbability = new double[maxLabel + 1][maxFeature + 1];

		//ラベルと特徴の出現回数を数える
		for(LabeledVector lv : trainingDataSet) {
			labelCount[lv.label]++;
			for(Entry<Integer, Double> entry : lv.featureVector.entrySet()) {
				if(entry.getValue() != 0.0) {	//０以外の値は全て１とみなす
					featureCount[lv.label][entry.getKey()]++;
				}
			}
		}
		//ラベルと特徴の出現確率を計算する（スムージング使用）
		for(int i = 1; i <= maxLabel; i++) {
			labelProbability[i] = (double)(labelCount[i] + alpha) / (trainingDataSet.size() + alpha * maxLabel);
			for(int j = 1; j <= maxFeature; j++) {
				featureProbability[i][j] = (double)(featureCount[i][j] + alpha) / (labelCount[i] + alpha * 2.0);
			}
		}
		return;
	}

	/*テストデータを分類する*/
	@Override
	public int classify(Map<Integer, Double> featureVector) {
		double maxProb = 0.0;
		int maxProbLabel = 0;

		for(int i = 1; i <= maxLabel; i++) {
			double prob = labelProbability[i];
			for(int j = 1; j <= maxFeature; j++) {
				Double d = featureVector.get(j);
				if(d != null && d != 0.0) {
					prob *= featureProbability[i][j];
				}else {
					prob *= (1 - featureProbability[i][j]);
				}
			}
			if(prob > maxProb) {
				maxProb = prob;
				maxProbLabel = i;
			}
		}
		return maxProbLabel;
	}
}
