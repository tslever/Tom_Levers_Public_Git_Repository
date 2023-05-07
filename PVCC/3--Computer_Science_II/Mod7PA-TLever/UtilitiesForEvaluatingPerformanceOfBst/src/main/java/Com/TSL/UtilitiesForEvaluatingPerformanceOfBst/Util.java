package Com.TSL.UtilitiesForEvaluatingPerformanceOfBst;


import java.util.HashSet;
import java.util.Iterator;


public class Util {
	
	public static int[] randomArray(int var0, int var1) {
		
		HashSet var2 = new HashSet();

		while (var2.size() < var0) {
			int var3 = (int) (Math.random() * (double) var1) + 1;
			var2.add(var3);
		}

		int[] var7 = new int[var0];
		int var4 = 0;

		Integer var6;
		for (Iterator var5 = var2.iterator(); var5.hasNext(); var7[var4++] = var6) {
			var6 = (Integer) var5.next();
		}

		return var7;
	}
}