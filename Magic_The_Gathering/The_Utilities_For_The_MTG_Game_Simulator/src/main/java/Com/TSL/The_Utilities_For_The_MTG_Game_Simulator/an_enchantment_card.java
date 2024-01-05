package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class an_enchantment_card extends a_nonland_card {

	private String Rarity;
	
	
	public an_enchantment_card(
		String The_Expansion_To_Use,
		a_mana_cost The_Mana_Cost_To_Use,
		String The_Name_To_Use,
		String The_Rarity_To_Use,
		ArrayList<String> The_Text_To_Use,
		String The_Type_To_Use
	) {
		
		super(The_Expansion_To_Use, The_Mana_Cost_To_Use, The_Name_To_Use, The_Text_To_Use, The_Type_To_Use);
		
		this.Rarity = The_Rarity_To_Use;
		
	}
}
