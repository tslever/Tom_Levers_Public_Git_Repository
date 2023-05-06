package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public abstract class a_card
{
	private String Expansion;
	private boolean Is_Playable;
	private ArrayList<a_permanent> List_Of_Mana_Contributing_Permanents;
	private a_mana_pool Mana_Subpool;
	private String Name;
	private String Type;
	private a_mana_cost Mana_Cost;
	
	protected a_card(String The_Expansion_To_Use, a_mana_cost The_Mana_Cost_To_Use, String The_Name_To_Use, String The_Type_To_Use) {
		this.Expansion = The_Expansion_To_Use;
		this.Is_Playable = false;
		this.List_Of_Mana_Contributing_Permanents = null;
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
	
	public void nullifies_its_mana_subpool() {
		this.Mana_Subpool = null;
	}
	
	public void nullifies_its_list_of_mana_contributing_permanents() {
		this.List_Of_Mana_Contributing_Permanents = null;
	}
	
	public a_mana_cost provides_its_mana_cost() {
		return this.Mana_Cost;
	}
	
	public String provides_its_name() {
		return this.Name;
	}
	
	
	public String provides_its_type() {
		return this.Type;
	}
	
	public void receives(ArrayList<a_permanent> The_List_Of_Mana_Contributing_Permanents_To_Use) {
		this.List_Of_Mana_Contributing_Permanents = The_List_Of_Mana_Contributing_Permanents_To_Use;
	}
	
	public void receives(a_mana_pool The_Mana_Subpool_To_Use) {
		this.Mana_Subpool = The_Mana_Subpool_To_Use;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
}
