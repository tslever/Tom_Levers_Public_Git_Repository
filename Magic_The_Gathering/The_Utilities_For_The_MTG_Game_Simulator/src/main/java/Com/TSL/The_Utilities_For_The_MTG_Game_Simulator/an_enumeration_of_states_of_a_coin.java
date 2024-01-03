package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.Random;


public enum an_enumeration_of_states_of_a_coin {

	HEADS,
	TAILS;
	
	
	public static an_enumeration_of_states_of_a_coin provides_a_state() {
		
		return an_enumeration_of_states_of_a_coin.values()[new Random().nextInt(an_enumeration_of_states_of_a_coin.values().length)];
	}
}
