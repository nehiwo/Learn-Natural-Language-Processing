package chapter2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

/*サポートベクタマシーン分類器*/

public class SVMClassifier implements Classifier {
	/*SVM分類器のパラメータ*/
	public svm_parameter parameter;
	/*SVM分類器のモデル*/
	public svm_model model;

	/*コンストラクタ：分類器パラメータの初期化を行う*/
	public SVMClassifier(){
		parameter = new svm_parameter();

		//各パラメータの値を適当に設定する
		//値を調整する場合は、まずcやkernel＿typeを変えてみると良い
		parameter.svm_type = svm_parameter.C_SVC;	//SVMの種類
		parameter.C = 1.0;	//制約を緩めた分類に用いるパラメータ
		parameter.kernel_type = svm_parameter.LINEAR;	//カーネルの種類
		parameter.gamma = 0;	//POLY,RBF,SIGMOIDで有効
		parameter.degree = 2;	//POLYにおける多項式の次数
		parameter.coef0 = 0;	//POLY,SIGMOIDにおける係数
		parameter.cache_size = 100;	//キャッシュサイズ
		parameter.eps = 1e-3;	//最適化計算停止条件
	}

	/*特徴ベクトルのデータ型変換*/
	public svm_node[] toSVMNodes(Map<Integer, Double> featureVector) {
		List<Integer> indexList = new ArrayList<Integer>(featureVector.keySet());
		Collections.sort(indexList);	//indexを小さい順に並べる
		svm_node[] nodes = new svm_node[indexList.size()];
		for(int i = 0; i < indexList.size(); i++) {
			nodes[i] = new svm_node();
			nodes[i].index = indexList.get(i);
			nodes[i].value = featureVector.get(indexList.get(i));
		}
		return nodes;
	}

	/*学習データセットを使って学習を行う*/
	@Override
	public void train(List<LabeledVector> trainingDataSet, int maxLabel, int maxFeature) {
		//学習データをLIBSVMのデータ構造へ変換する
		svm_problem prob = new svm_problem();
		prob.l = trainingDataSet.size();
		prob.x = new svm_node[prob.l][];
		prob.y = new double[prob.l];

		for(int i = 0; i < trainingDataSet.size(); i++) {
			prob.x[i] = toSVMNodes(trainingDataSet.get(i).featureVector);
			prob.y[i] = (double)trainingDataSet.get(i).label;
		}

		//学習実行
		model = svm.svm_train(prob, parameter);
	}

	/*テストデータを分類する*/
	@Override
	public int classify(Map<Integer, Double> featureVector) {
		svm_node[] nodes = toSVMNodes(featureVector);
		double x = svm.svm_predict(model,  nodes);
		return (int)x;
	}

}
