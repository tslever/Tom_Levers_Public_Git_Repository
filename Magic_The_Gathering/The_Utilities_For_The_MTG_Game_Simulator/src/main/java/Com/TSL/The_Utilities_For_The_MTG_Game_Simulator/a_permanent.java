package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public abstract class a_permanent {

	private String Name;
	private ArrayList<a_mana_ability> List_Of_Mana_Abilities;
	private ArrayList<a_nonmana_activated_ability> List_Of_Nonmana_Activated_Abilities;
	private boolean Tapped_Status;
	private a_player Player;
	
	public a_permanent(String The_Name_To_Use, a_player The_Player_To_Use) {
		this.Name = The_Name_To_Use;
		this.List_Of_Mana_Abilities = new ArrayList<a_mana_ability>();
		this.List_Of_Nonmana_Activated_Abilities = new ArrayList<a_nonmana_activated_ability>();
		this.Tapped_Status = false;
		this.Player = The_Player_To_Use;
	}
	
	public boolean is_tapped() {
		return this.Tapped_Status;
	}
	
	public String provides_its_name() {
		return this.Name;
	}
	
	public ArrayList<a_mana_ability> provides_a_list_of_available_mana_abilities() {
		ArrayList<a_mana_ability> The_List_Of_Available_Mana_Abilities = new ArrayList<>();
		for (a_mana_ability The_Mana_Ability : this.List_Of_Mana_Abilities) {
			if (!The_Mana_Ability.requires_tapping() || !this.Tapped_Status) {
				The_List_Of_Available_Mana_Abilities.add(The_Mana_Ability);
			}
		}
		return The_List_Of_Available_Mana_Abilities;
	}
	
	public ArrayList<a_nonmana_activated_ability> provides_its_list_of_nonmana_activated_abilities() {
		return this.List_Of_Nonmana_Activated_Abilities;
	}
	
	public void receives(a_mana_ability The_Mana_Ability_To_Use) {
		this.List_Of_Mana_Abilities.add(The_Mana_Ability_To_Use);
	}
	
	public void taps() {
		System.out.println(this.Name + " has tapped.");
		this.Tapped_Status = true;
	}
	
	public void untaps() {
		System.out.println(this.Name + " has untapped.");
		this.Tapped_Status = false;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
	
	public a_player player() {
		return this.Player;
	}
}