package lattice_utilities;

public abstract class Lattice {
	
	
	protected int height;
	protected int width;
	private int[][] cells;
	private String[] icons;
	
	
	public Lattice(int heightToUse, int widthToUse) {
		
		this.height = heightToUse;
		this.width = widthToUse;
		this.cells = new int[heightToUse][widthToUse];
		this.icons = new String[] {"-", "1"};
		
	}
	
	
	public void setCellIdentifiedBy(int row, int column) {
		
		this.cells[row][column] = 1;
		
	}
	
	
	public void display() {
		
		int j;
		for (int i = 0; i < this.height; i++) {
			
			for (j = 0; j < this.width-1; j++) {
				System.out.print(this.icons[cells[i][j]] + " ");
			}
			System.out.println(this.icons[cells[i][this.width-1]]);
			
		}
		
	}
	
	
	public int height() {
		
		return this.height;
		
	}
	
	
	public int width() {
		
		return this.width;
		
	}
	
	
	public int valueOfCellIdentifiedBy(int row, int column) {
		
		return this.cells[row][column];
		
	}
	

}