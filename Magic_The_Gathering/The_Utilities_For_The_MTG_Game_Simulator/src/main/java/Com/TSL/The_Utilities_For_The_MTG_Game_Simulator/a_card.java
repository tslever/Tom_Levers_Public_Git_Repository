package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public abstract class a_card
{
	private String Expansion;
	private boolean Is_Playable;
	private ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana;
	private String Name;
	private String Type;
	private a_mana_pool Mana_Cost;
	
	protected a_card(String The_Expansion_To_Use, a_mana_pool The_Mana_Cost_To_Use, String The_Name_To_Use, String The_Type_To_Use) {
		this.Expansion = The_Expansion_To_Use;
		this.Is_Playable = false;
		this.List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana = null;
		this.Mana_Cost = The_Mana_Cost_To_Use;
		this.Name = The_Name_To_Use;
		this.Type = The_Type_To_Use;
	}

	public void becomes_not_playable() {
		this.Is_Playable = false;
	}
	
	public void becomes_playable() {
		this.Is_Playable = true;
	}
	
	public boolean is_playable() {
		return this.Is_Playable;
	}
	
	public void nullifies_its_list_of_sufficient_combinations_of_configurations_of_a_permanent_to_contribute_mana() {
		this.List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana = null;
	}
	
	public a_mana_pool provides_its_mana_cost() {
		return this.Mana_Cost;
	}
	
	public String provides_its_name() {
		return this.Name;
	}
	
	
	public String provides_its_type() {
		return this.Type;
	}
	
	public void receives(ArrayList<ArrayList<a_configuration_of_a_permanent_to_contribute_mana>> The_List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana_To_Use) {
		this.List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana = The_List_Of_Sufficient_Combinations_Of_Configurations_Of_A_Permanent_To_Contribute_Mana_To_Use;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
}
