package midi_parsing_utilities;

public class TickCommandKeyAndVelocity {

	private int tick;
	private int command;
	private int key;
	private int velocity;
	private boolean usedToInsertSinusoidInTimeSeries;
	
	public TickCommandKeyAndVelocity(int tickToUse, int commandToUse, int keyToUse, int velocityToUse) {
		
		this.tick = tickToUse;
		this.command = commandToUse;
		this.key = keyToUse;
		this.velocity = velocityToUse;
		this.usedToInsertSinusoidInTimeSeries = false;
		
	}
	
	public int getTick() {
		return this.tick;
	}
	
	public int getCommand() {
		return this.command;
	}
	
	public int getKey() {
		return this.key;
	}
	
	public int getVelocity() {
		return this.velocity;
	}
	
	public boolean getUsedToInsertSinusoidInTimeSeries() {
		return this.usedToInsertSinusoidInTimeSeries;
	}
	
	public void setUsedToInsertSinusoidInTimeSeries() {
		this.usedToInsertSinusoidInTimeSeries = true;
	}
}