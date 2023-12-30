package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public class a_sorcery_card extends a_card {

	private String Rarity;
	private String[] Text;
	
	public a_sorcery_card(
		String The_Expansion_To_Use,
		a_mana_pool The_Mana_Cost_To_Use,
		String The_Name_To_Use,
		String The_Rarity_To_Use,
		String[] The_Text_To_Use,
		String The_Type_To_Use
	) {
		super(The_Expansion_To_Use, The_Mana_Cost_To_Use, The_Name_To_Use, The_Type_To_Use);
		
		this.Rarity = The_Rarity_To_Use;
		this.Text = The_Text_To_Use;
	}
	
}
