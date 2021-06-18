package chapter2;

import java.util.List;
import java.util.Map;

/*分類器を表すインターフェース*/

public interface Classifier {
	/*学習データセットを使って学習を行う*/
	public void train(List<LabeledVector> trainingDataSet, int maxLabel, int maxFeature);

	/*テストデータを分類する*/
	public int classify(Map<Integer, Double> featureVector);
}
