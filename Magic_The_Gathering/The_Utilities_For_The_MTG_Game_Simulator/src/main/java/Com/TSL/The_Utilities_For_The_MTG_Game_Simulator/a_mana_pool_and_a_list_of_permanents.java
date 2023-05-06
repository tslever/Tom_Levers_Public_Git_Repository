package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_mana_pool_and_a_list_of_permanents {
	
	private a_mana_pool Mana_Pool;
	private ArrayList<a_permanent> List_Of_Permanents;
	
	public a_mana_pool_and_a_list_of_permanents(a_mana_pool The_Mana_Pool_To_Use, ArrayList<a_permanent> The_List_Of_Permanents_To_Use) {
		this.Mana_Pool = The_Mana_Pool_To_Use;
		this.List_Of_Permanents = The_List_Of_Permanents_To_Use;
	}
	
	public a_mana_pool provides_its_mana_pool() {
		return this.Mana_Pool;
	}
	
	public ArrayList<a_permanent> provides_its_list_of_permanents() {
		return this.List_Of_Permanents;
	}
}