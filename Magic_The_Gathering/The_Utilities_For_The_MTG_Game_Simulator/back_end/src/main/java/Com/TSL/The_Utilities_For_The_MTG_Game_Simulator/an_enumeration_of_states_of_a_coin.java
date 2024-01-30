package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.Random;

public enum an_enumeration_of_states_of_a_coin {

	HEADS,
	TAILS;
	
	public static an_enumeration_of_states_of_a_coin provides_a_random_state() {
		int The_Number_Of_Values = an_enumeration_of_states_of_a_coin.values().length;
		int The_Random_Index_Of_A_State = new Random().nextInt(The_Number_Of_Values);
		return an_enumeration_of_states_of_a_coin.values()[The_Random_Index_Of_A_State];
	}
}