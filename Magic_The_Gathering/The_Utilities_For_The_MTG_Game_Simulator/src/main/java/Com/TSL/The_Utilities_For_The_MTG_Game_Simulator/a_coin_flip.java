package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.Random;


public enum a_coin_flip {

	HEADS,
	TAILS;
	
	
	public static a_coin_flip provides_a_random_value() {
		
		return a_coin_flip.values()[new Random().nextInt(a_coin_flip.values().length)];
	}
}
