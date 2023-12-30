package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public class a_creature_card extends a_nonland_card
{
	private int Power;
	private String Rarity;
	private String Subtype;
	private String[] Text;
	private int Toughness;
	
	
	public a_creature_card(
		String The_Expansion_To_Use,
		a_mana_pool The_Mana_Cost_To_Use,
		String The_Name_To_Use,
		int The_Power_To_Use,
		String The_Rarity_To_Use,
		String The_Subtype_To_Use,
		String[] The_Text_To_Use,
		int The_Toughness_To_Use,
		String The_Type_To_Use
	)
	{
		super(The_Expansion_To_Use, The_Mana_Cost_To_Use, The_Name_To_Use, The_Type_To_Use);
		
		this.Power = The_Power_To_Use;
		this.Rarity = The_Rarity_To_Use;
		this.Subtype = The_Subtype_To_Use;
		this.Text = The_Text_To_Use;
		this.Toughness = The_Toughness_To_Use;
	}
}
