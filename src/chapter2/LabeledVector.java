package chapter2;

import java.util.Map;

/*個々の学習データ（特徴ベクトル　＋　正解ラベル）を表すクラス*/

public class LabeledVector {
	/*データの名前*/
	public String name;
	/*特徴ベクトル*/
	public Map<Integer, Double> featureVector;
	/*正解ラベル*/
	public int label;
}
