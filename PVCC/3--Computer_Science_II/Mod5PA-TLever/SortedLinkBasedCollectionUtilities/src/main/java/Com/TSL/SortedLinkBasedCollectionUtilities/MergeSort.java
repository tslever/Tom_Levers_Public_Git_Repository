package Com.TSL.SortedLinkBasedCollectionUtilities;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-03
*
* MergeSort.java: MergeSort class
*/

public class MergeSort<T>{
   public void mergeSort(T[] arr){
      mergeSortRec(arr, 0, arr.length-1);
   }
   private void mergeSortRec(T[] arr, int first, int last){
      if(first<last){
         int mid=(first+last)/2;
         mergeSortRec(arr, first, mid);
         mergeSortRec(arr, mid+1,last);
         merge(arr, first, mid, mid+1, last);
      }
   }
   private void merge(T[] arr, int leftFirst, int leftLast, int rightFirst, int rightLast){
      T[] aux=(T[])(new Object[arr.length]);//extra space, this is downside of this algorithm
      int index=leftFirst;
      int saveFirst=leftFirst;
      while(leftFirst<=leftLast && rightFirst <=rightLast){
         if(((Comparable)(arr[leftFirst])).compareTo(arr[rightFirst])<=0){
            aux[index++]=arr[leftFirst++];
         }else{
            aux[index++]=arr[rightFirst++];
         }
      }
      while(leftFirst<=leftLast){
         aux[index++]=arr[leftFirst++];
      }
      while(rightFirst<=rightLast)
         aux[index++]=arr[rightFirst++];
      for(index=saveFirst; index<=rightLast; index++)
         arr[index]=aux[index];
   }
}