package chapter2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/*教師あり学習用データセット*/

public class LearningDataSet {
	/*データセット本体*/
	public List<LabeledVector> labeledVectorList;
	/*ラベルの最大値*/
	public int maxLabel;
	/*特徴のインデックスの最大値*/
	public int maxFeature;

	/*ファイルからデータを読み込む*/
	public static LearningDataSet readFromFile(String fileName) {
		LearningDataSet dataSet = new LearningDataSet();
		dataSet.labeledVectorList = new ArrayList<LabeledVector>();
		dataSet.maxLabel = 0;	//正解ラベルは１以上と仮定
		dataSet.maxFeature = 0;	//特徴インデックスは１以上と仮定

		try {
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			String line;
			while((line = br.readLine()) != null) {
				LabeledVector lv = new LabeledVector();
				String[] split1 = line.split("[ \t]+");	//空白またはタブで分割
				lv.label = Integer.parseInt(split1[0]);
				if(lv.label > dataSet.maxLabel) {
					dataSet.maxLabel = lv.label;
				}
				lv.name = split1[1];
				lv.featureVector = new HashMap<Integer, Double>();
				for(int i = 2; i < split1.length; i++) {
					String[] split2 = split1[i].split(":");
					int feature = Integer.parseInt(split2[0]);
					double value = Double.parseDouble(split2[1]);
					lv.featureVector.put(feature, value);
					if(feature > dataSet.maxFeature) {
						dataSet.maxFeature = feature;
					}
				}
				dataSet.labeledVectorList.add(lv);
			}
			br.close();
		}catch(IOException ex) {
			ex.printStackTrace();
		}
		return dataSet;
	}
}
