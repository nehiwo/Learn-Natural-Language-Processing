package chapter1;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
*青空文庫のテキストファイルをコーパス用のテキスト形式に変換する
*/

public class AozoraFilter{

	/*コンストラクタ*/
	public AozoraFilter() {

	}

	/*文字列からルビと注記を除去する*/
	String deleteRuby(String s) {
		char[] chars = s.toCharArray();
		StringBuffer sb = new StringBuffer();
		boolean inRuby = false;
		int inComment = 0;

		for(int i = 0; i < chars.length; i++) {
			if(chars[i] == '《') {
				if(inRuby) {
					System.out.println("エラー（’《’の重複）:" + s);
					System.exit(1);
				}else {
					inRuby = true;
				}
			}else if(chars[i] == '》') {
				if(inRuby) {
					inRuby = false;
				}else {
					System.out.println("エラー（’《’）の不存在:" + s);
					System.exit(1);
				}
			}else if(chars[i] == '［') {
				inComment++;
			}else if(chars[i] == '］') {
				if(inComment > 0) {
					inComment--;
				}else {
					System.out.println("エラー（’［’の不足）:" + s);
					System.exit(1);
				}
			}else if(chars[i] == '｜'){
				continue;
			}else {
				if(!inRuby && inComment == 0) {
					sb.append(chars[i]);
				}
			}
		}
		if(inRuby) {
			System.out.println("エラー（’》’の不存在）:" + s);
			System.exit(1);
		}
		if(inComment > 0) {
			System.out.println("エラー（’］’の不足）:" + s);
			System.exit(1);
		}
		return sb.toString();
	}

	/*文字列の先頭にある空白を除去する*/
	String deleteSpaces(String s) {
		int i;
		for(i = 0; i < s.length(); i++) {
			if(s.charAt(i) != ' ' && s.charAt(i) != '　' && s.charAt(i) != '\t') {
				break;
			}
		}
		return s.substring(i);
	}

	/*行を文単位で分割して出力し、余った文字列を返す*/
	String outputSentences(String line, PrintWriter pw, String line2) {
		int index = 0;
		while(index < line.length()) {
			int index2 = line.indexOf("。", index);
			if(index2 != -1) {
				if(index2 == line.length() - 2 && line.charAt(line.length() - 1) == '」'){
					index2++;	//行末の”。」”はまとめて扱う
				}
				String s = line2 + line.substring(index, index2 + 1);
				line2 = "";
				pw .println(s);
				index = index2 + 1;
			}else {
				line2 += line.substring(index);
				break;
			}
		}

		if(line2.length() > 0 && line2.charAt(line2.length() - 1) == '」') {
			pw.println(line2);	//会話文として出力する
			line2 = "";
		}
		if(line2.length() < 10) {
			line2 = "";//節見出しとして無視する
		}
		return line2;
	}

	/*青空文庫のテキストファイルをコーパス用のテキスト形式に変換する
	 * 引数はファイル名から拡張子（＂.text")を除いた文字列*/
	public void filter(String fileName) {
		String title = null;	//作品名
		String author = null;	//著者名
		int countForHeader = 0;	//ヘッダ範囲判定のために”----”の数を数える
		int countForFooter = 0;	//フッタ範囲判定のために空行の数を数える
		String resetOfLine = "";	//最後に読んだ行の末尾の未出力部分

		try {
			BufferedReader br = new BufferedReader(new FileReader(fileName + ".txt"));
			PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(fileName + "_filtered.txt")));

			title = br.readLine();
			author = br.readLine();
			System.out.println(title);
			System.out.println(author);

			String line;
			while((line = br.readLine()) != null) {
				//ヘッダとフッタのための処理
				if(line.startsWith("----")) {
					countForHeader++;
					continue;
				}
				if(countForHeader < 2) {
					continue;	//ヘッダが終わるまで読み飛ばす
				}
				if(line.contentEquals("")) {
					countForFooter++;
					continue;
				}
				if(countForFooter >= 3 && line.startsWith("底本")) {
					break;	//フッタが始まったので読み飛ばす
				}
				countForFooter = 0;

				//本文行の整形と出力
				String line2 = deleteSpaces(line);	//先頭の空白の除去
				String line3 = deleteRuby(line2);	//ルビ・注記の削除
				resetOfLine = outputSentences(line3, pw, resetOfLine);	//文単位で出力
			}
			if(!resetOfLine.equals("")) {
				pw.println(resetOfLine);
			}
			br.close();
			pw.close();
		}catch(Exception ex) {
			ex.printStackTrace();
		}
	}

	/*変換の実行*/
	public static void main(String[] args) {
		AozoraFilter af = new AozoraFilter();
		af.filter("kokoro");
	}
}