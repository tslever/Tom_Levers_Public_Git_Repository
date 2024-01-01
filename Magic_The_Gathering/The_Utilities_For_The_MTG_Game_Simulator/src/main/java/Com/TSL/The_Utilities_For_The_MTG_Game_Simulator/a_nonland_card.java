package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_nonland_card extends a_card {

	private ArrayList<ArrayList<a_mana_ability>> List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities;
	private a_mana_pool Mana_Cost;
	
	protected a_nonland_card(String The_Expansion_To_Use, a_mana_pool The_Mana_Cost_To_Use, String The_Name_To_Use, ArrayList<String> The_Text_To_Use, String The_Type_To_Use) {
		super(The_Expansion_To_Use, The_Name_To_Use, The_Text_To_Use, The_Type_To_Use);
		this.List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities = new ArrayList<ArrayList<a_mana_ability>>();
		this.Mana_Cost = The_Mana_Cost_To_Use;
	}
	
	public a_mana_pool provides_its_mana_cost() {
		return this.Mana_Cost;
	}
	
	public ArrayList<ArrayList<a_mana_ability>> provides_its_list_of_sufficient_combinations_of_available_mana_abilities() {
		return this.List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities;
	}
	
	public void receives(ArrayList<ArrayList<a_mana_ability>> The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities_To_Use) {
		this.List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities = The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities_To_Use;
	}
	
}