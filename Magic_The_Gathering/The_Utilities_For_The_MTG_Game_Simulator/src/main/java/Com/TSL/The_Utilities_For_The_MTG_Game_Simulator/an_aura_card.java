package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public class an_aura_card extends an_enchantment_card {

	private String Subtype;
	
	public an_aura_card(
		String The_Expansion_To_Use,
		a_mana_cost The_Mana_Cost_To_Use,
		String The_Name_To_Use,
		String The_Rarity_To_Use,
		String The_Subtype_To_Use,
		String[] The_Text_To_Use,
		String The_Type_To_Use
	) {
		
		super(The_Expansion_To_Use, The_Mana_Cost_To_Use, The_Name_To_Use, The_Rarity_To_Use, The_Text_To_Use, The_Type_To_Use);
		
		this.Subtype = The_Subtype_To_Use;
	}
}