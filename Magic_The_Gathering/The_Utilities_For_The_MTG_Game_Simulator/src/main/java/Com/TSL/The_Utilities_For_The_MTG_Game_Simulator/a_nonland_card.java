package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_nonland_card extends a_card {

	private ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana;
	private a_mana_pool Mana_Cost;
	
	protected a_nonland_card(String The_Expansion_To_Use, a_mana_pool The_Mana_Cost_To_Use, String The_Name_To_Use, String The_Type_To_Use) {
		super(The_Expansion_To_Use, The_Name_To_Use, The_Type_To_Use);
		this.List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana = new ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>>();
		this.Mana_Cost = The_Mana_Cost_To_Use;
	}
	
	public a_mana_pool provides_its_mana_cost() {
		return this.Mana_Cost;
	}
	
	public ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> provides_its_list_of_sufficient_combinations_of_configurations_of_a_permanent_to_contribute_mana() {
		return this.List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana;
	}
	
	public void receives(ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> The_List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana_To_Use) {
		this.List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana = The_List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana_To_Use;
	}
	
}
