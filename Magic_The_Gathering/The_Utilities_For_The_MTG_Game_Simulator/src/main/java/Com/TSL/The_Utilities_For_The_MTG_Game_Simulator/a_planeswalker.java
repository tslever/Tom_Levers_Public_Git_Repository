package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


public class a_planeswalker extends a_permanent {
	
	private int loyalty;
	private int Combat_Damage;
	
	public a_planeswalker(String The_Name, boolean The_Tapped_Status, a_player The_Player) {
		super(The_Name, The_Player);
	}
	
	public void receives_combat_damage(int The_Combat_Damage) {
		this.Combat_Damage = The_Combat_Damage;
	}
}