/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.serverutilities.productutilities;

/**
 *
 * @author thoma
 */
public class Product {
    
    private String name;
    private String pathToImageOfClosedProduct;
    private String pathToImageOfOpenProduct;
    private String description;
    private String story;
    private String productId;
    
    public Product() { };
    
    public void setName(String nameToUse) {
        this.name = nameToUse;
    }

    public void setPathToImageOfClosedProduct(
        String pathToImageOfClosedProductToUse) {
        this.pathToImageOfClosedProduct = pathToImageOfClosedProductToUse;
    }
    
    public void setPathToImageOfOpenProduct(
        String pathToImageOfOpenProductToUse) {
        this.pathToImageOfOpenProduct = pathToImageOfOpenProductToUse;
    }
    
    public void setDescription(String descriptionToUse) {
        this.description = descriptionToUse;
    }
    
    public void setStory(String storyToUse) {
        this.story = storyToUse;
    }
    
    public void setProductId(String productIdToUse) {
        this.productId = productIdToUse;
    }
    
    @Override
    public String toString() {
        return
            "{" +
                "\"name\": \"" + this.name + "\", " +
                "\"image_closed\": \"" + this.pathToImageOfClosedProduct +
                    "\", " +
                "\"image_open\": \"" + this.pathToImageOfOpenProduct + "\", " +
                "\"description\": \"" + this.description + "\", " +
                "\"story\": \"" + this.story + "\", " +
                "\"productId\": \"" + this.productId + "\"" +
            "}";
    }
    
}
