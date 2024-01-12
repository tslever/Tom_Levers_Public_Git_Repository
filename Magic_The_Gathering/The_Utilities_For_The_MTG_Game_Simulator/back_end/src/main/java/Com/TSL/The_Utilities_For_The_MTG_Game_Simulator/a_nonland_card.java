package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_nonland_card extends a_card {

	private ArrayList<ArrayList<a_mana_ability>> List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This;
	private a_mana_cost Mana_Cost;
	
	protected a_nonland_card(String The_Expansion_To_Use, a_mana_cost The_Mana_Cost_To_Use, String The_Name_To_Use, ArrayList<String> The_Text_To_Use, String The_Type_To_Use) {
		super(The_Expansion_To_Use, The_Name_To_Use, The_Text_To_Use, The_Type_To_Use);
		this.List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This = new ArrayList<ArrayList<a_mana_ability>>();
		this.Mana_Cost = The_Mana_Cost_To_Use;
	}
	
	public a_mana_cost mana_cost() {
		return this.Mana_Cost;
	}
	
	public ArrayList<ArrayList<a_mana_ability>> list_of_combinations_of_available_mana_abilities_sufficient_to_play_this() {
		return this.List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This;
	}
	
	public void receives(ArrayList<ArrayList<a_mana_ability>> The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This_To_Use) {
		this.List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This = The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This_To_Use;
	}
}