package chapter2;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

/*ニューラルネットワーク分類器*/

public class NeuralClassifier implements Classifier{
	/*ラベルの最大値*/
	public int maxLabel;
	/*特徴のインデックスの最大値*/
	public int maxFeature;
	/*中間層のノード数*/
	public int numHiddenUnits;
	/*入力層から中間層のリンク重み*/
	public double[][] w1;
	/*中間層から出力層のリンク重み*/
	public double[][] w2;
	/*学習率*/
	public double eta;
	/*最大エポック数*/
	public int maxEpoch;
	/*誤差の閾値*/
	public double threshold;

	/*学習データセットを使って学習を行う*/
	@Override
	public void train(List<LabeledVector> trainingDataSet, int maxLabel, int maxFeature) {
		this.maxLabel = maxLabel;
		this.maxFeature = maxFeature;
		Random random = new Random();

		//辺の重みを乱数で初期化する
		w1 = new double[numHiddenUnits + 1][maxFeature + 1];
		for(int i = 1; i <= numHiddenUnits; i++) {
			for(int j = 0; j <= maxFeature; j++) {
				w1[i][j] = random.nextDouble() - 0.5;
			}
		}
		w2 = new double[maxLabel + 1][numHiddenUnits + 1];
		for(int i = 1; i <= maxLabel; i++) {
			for(int j = 0; j <= numHiddenUnits; j++) {
				w2[i][j] = random.nextDouble() - 0.5;
			}
		}

		//入力層ユニット
		double[] inputUnits = new double[maxFeature + 1];
		//中間層ユニット
		double[] hiddenUnits = new double[numHiddenUnits + 1];
		//出力層ユニット
		double[] outputUnits = new double[maxLabel + 1];

		//各出力ユニットが出力すべき値
		double[] answer = new double[maxLabel + 1];
		//中間層における誤差の大きさを表す量
		double[] delta1 = new double[numHiddenUnits + 1];
		//出力層における誤差の大きさを表す量
		double[] delta2 = new double[maxLabel + 1];

		//学習データ全体にわたる学習をmaxEpoch回繰り返す
		for(int epoch = 0; epoch <= maxEpoch; epoch++) {
			double err = 0.0;	//平均二乗誤差
			for(LabeledVector lv : trainingDataSet) {
				//入力層ユニットへの入力値を設定
				for(Entry<Integer, Double> entry : lv.featureVector.entrySet()) {
					inputUnits[entry.getKey()] = entry.getValue();
				}
				inputUnits[0] = 1.0;	//バイアス項に相当

				//入力層から中間層へ
				for(int i = 1; i <= numHiddenUnits; i++) {
					double u = 0.0;
					for(int j = 0; j <= maxFeature; j++) {
						u += w1[i][j] * inputUnits[j];
					}
					hiddenUnits[i] = sigmoid(u);
				}
				hiddenUnits[0] = 1.0;	//バイアス項に相当

				//中間層から出力層へ
				for(int i = 1; i <= maxLabel; i++) {
					double u = 0.0;
					for(int j = 0; j <= numHiddenUnits; j++) {
						u += w2[i][j] * hiddenUnits[j];
					}
					outputUnits[i] = sigmoid(u);
				}

				//誤差逆伝播学習
				for(int i = 1; i <= maxLabel; i++) {
					answer[i] = (i == lv.label) ? 1.0 : 0.0;
				}
				for(int i = 1; i <= maxLabel; i++) {
					delta2[i] = (outputUnits[i] - answer[i]) * (1.0 - outputUnits[i] * outputUnits[i]);
					err += (outputUnits[i] - answer[i]) * (outputUnits[i] - answer[i]);
				}
				for(int i = 1; i <= numHiddenUnits; i++) {
					delta1[i] = 0.0;
					for(int j = 0; j <= numHiddenUnits; j++) {
						delta1[i] += w2[j][i] * delta2[j];
					}
					delta1[i] *= ((1 - hiddenUnits[i]) * hiddenUnits[i]);
				}

				for(int i = 1; i <= maxLabel; i++) {
					for(int j = 0; j <= numHiddenUnits; j++) {
						w2[i][j] -= eta * delta2[i] * hiddenUnits[j];
					}
				}
				for(int i = 1; i <= numHiddenUnits; i++) {
					for(int j = 0; j <= maxFeature; j++) {
						w1[i][j] -= eta * delta1[i] * inputUnits[j];
					}
				}
			}
			err /= 2.0;
			err /= trainingDataSet.size();
			System.out.println("epoch = " + epoch + "\terr = " + err);
			if(err < threshold) {
				break;
			}
		}
		//終了処理
		System.out.println("trainig done");
	}

	/*シグモイド関数*/
	public double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	/*テストデータを分類する*/
	@Override
	public int classify(Map<Integer, Double> featureVector) {
		//入力層ユニット
		double[] inputUnits = new double[maxFeature + 1];
		//中間層ユニット
		double[] hiddenUnits = new double[numHiddenUnits + 1];
		//出力層ユニット
		double[] outputUnits = new double[maxLabel + 1];

		//入力層ユニットへの入力値を設定
		for(Entry<Integer, Double> entry : featureVector.entrySet()) {
			inputUnits[entry.getKey()] = entry.getValue();
		}
		inputUnits[0] = 1.0;	//バイアス項に相当

		//入力層から中間層へ
		for(int i = 1; i <= numHiddenUnits; i++) {
			double u = 0.0;
			for(int j = 0; j <= maxFeature; j++) {
				u += w1[i][j] * inputUnits[j];
			}
			hiddenUnits[i] = sigmoid(u);
		}
		hiddenUnits[0] = 1.0;	//バイアス項に相当

		//中間層から出力層へ
		for(int i = 1; i <= maxLabel; i++) {
			double u = 0.0;
			for(int j = 0; j <= numHiddenUnits; j++) {
				u += w2[i][j] * hiddenUnits[j];
			}
			outputUnits[i] = sigmoid(u);
		}

		double maxProb = 0.0;
		int maxProbLabel = 0;

		for(int i = 1; i <= maxLabel; i++) {
			if(outputUnits[i] > maxProb) {
				maxProb = outputUnits[i];
				maxProbLabel = i;
			}
		}
		return maxProbLabel;	//出力値が最大のラベルを返す
	}

}
