package Com.TSL.ElasticsearchIndexerAndQuerier;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.io.StringReader;
import java.util.List;

import org.apache.http.HttpHost;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.CredentialsProvider;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.apache.http.impl.nio.client.HttpAsyncClientBuilder;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestClientBuilder;
import org.elasticsearch.client.RestClientBuilder.HttpClientConfigCallback;
import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.DeleteRequest;
import co.elastic.clients.elasticsearch.core.DeleteResponse;
import co.elastic.clients.elasticsearch.core.IndexRequest;
import co.elastic.clients.elasticsearch.core.IndexResponse;
import co.elastic.clients.elasticsearch.core.SearchRequest;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.elasticsearch.indices.CreateIndexRequest;
import co.elastic.clients.elasticsearch.indices.CreateIndexResponse;
import co.elastic.clients.elasticsearch.indices.DeleteIndexRequest;
import co.elastic.clients.elasticsearch.indices.DeleteIndexResponse;
import co.elastic.clients.elasticsearch.indices.PutMappingRequest;
import co.elastic.clients.elasticsearch.indices.PutMappingResponse;
import co.elastic.clients.elasticsearch.indices.RefreshResponse;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.ElasticsearchTransport;
import co.elastic.clients.transport.rest_client.RestClientTransport;

public class ElasticsearchIndexerAndQuerierTest {

    @Test
    public void testElasticsearchIndexerAndQuerier() throws IOException {
    
        // Create a credentials provider.
        final CredentialsProvider credentialsProvider = new BasicCredentialsProvider();
        credentialsProvider.setCredentials(AuthScope.ANY, new UsernamePasswordCredentials("elastic", "password"));
        
        // Create a REST client.
        RestClientBuilder restClientBuilder = RestClient.builder(new HttpHost("localhost", 9200));
        HttpClientConfigCallback httpClientConfigCallback = new HttpClientConfigCallback() {
            @Override
            public HttpAsyncClientBuilder customizeHttpClient(HttpAsyncClientBuilder httpClientBuilder) {
                return httpClientBuilder.setDefaultCredentialsProvider(credentialsProvider);
            }
        };
        restClientBuilder.setHttpClientConfigCallback(httpClientConfigCallback);
        RestClient restClient = restClientBuilder.build();
    
        ElasticsearchTransport elasticsearchTransport = new RestClientTransport(restClient, new JacksonJsonpMapper());
    
        ElasticsearchClient client = new ElasticsearchClient(elasticsearchTransport);
        
        try {
        
            // Create index commit.
            CreateIndexRequest.Builder createIndexRequestBuilder = new CreateIndexRequest.Builder();
            createIndexRequestBuilder.index("commit");
            CreateIndexRequest createIndexRequest = createIndexRequestBuilder.build();
            CreateIndexResponse createIndexResponse = client.indices().create(createIndexRequest);
            System.out.println(createIndexResponse);
            
            // Store and index commit-dateTime strings as dates and commit-dangeTimeRange strings as date ranges.
            PutMappingRequest.Builder putMappingRequestBuilder = new PutMappingRequest.Builder();
            putMappingRequestBuilder.index("commit");
            String mappings =
                "{\n" +
                "    \"properties\": {\n" +
                "        \"id\": {\n" +
                "            \"type\": \"text\"\n" +
                "        },\n" +
                "        \"name\": {\n" +
                "            \"type\": \"text\"\n" +
                "        },\n" +
                "        \"dateTimeRange\": {\n" +
                "            \"type\": \"date_range\",\n" +
                "            \"format\": \"uuuu-MM-dd[['T'][ ]HH:mm[:ss[[.][,][SSS][SS][S]]]][XXXXX][XXX][X]\"\n" +
                "        }\n" +
                "    }\n" +
                "}";
            putMappingRequestBuilder.withJson(new StringReader(mappings));
            PutMappingRequest putMappingRequest = putMappingRequestBuilder.build();
            PutMappingResponse putMappingResponse = client.indices().putMapping(putMappingRequest);
            System.out.println(putMappingResponse);
            
            // Index a commit.
            IndexRequest.Builder<Commit> indexRequestBuilder = new IndexRequest.Builder<>();
            indexRequestBuilder.index("commit");
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode dateTimeRange = objectMapper.readValue("{\"gte\":\"2022-07-27T00:00:00.000Z\",\"lte\":\"2022-07-27T23:59:59.999Z\"}", JsonNode.class);
            Commit commit = new Commit("commit20220727", "Commit 2022-07-27", dateTimeRange);
            indexRequestBuilder.id(commit.getId());
            indexRequestBuilder.document(commit);
    //        String indexRequestBody =
    //            "{\n" +
    //            "    \"id\": \"" + commit.id + "\",\n" +
    //            "    \"name\": \"" + commit.name + "\",\n" +
    //            "    \"dateTimeRange\": {\n" +
    //            "        \"gte\": \"2022-07-27T00:00:00.000Z\",\n" + 
    //            "        \"lte\": \"2022-07-27T23:59:59.999Z\"\n" +
    //            "    }\n" +
    //            "}";
    //        indexRequestBuilder.withJson(new StringReader(indexRequestBody));
            IndexResponse indexResponse = client.index(indexRequestBuilder.build());
            System.out.println(indexResponse);
            
            // Index a commit.
            indexRequestBuilder = new IndexRequest.Builder<>();
            indexRequestBuilder.index("commit");
            dateTimeRange = objectMapper.readValue("{\"gte\":\"2022-07-28T00:00:00.000Z\",\"lte\":\"2022-07-28T23:59:59.999Z\"}", JsonNode.class);
            commit = new Commit("commit20220728", "Commit 2022-07-28", dateTimeRange);
            indexRequestBuilder.id(commit.getId());
            indexRequestBuilder.document(commit);
    //        indexRequestBody =
    //            "{\n" +
    //            "    \"id\": \"" + commit.id + "\",\n" +
    //            "    \"name\": \"" + commit.name + "\",\n" +
    //            "    \"dateTimeRange\": {\n" +
    //            "        \"gte\": \"2022-07-28T00:00:00.000Z\",\n" + 
    //            "        \"lte\": \"2022-07-28T23:59:59.999Z\"\n" +
    //            "    }\n" +
    //            "}";
    //        indexRequestBuilder.withJson(new StringReader(indexRequestBody));
            indexResponse = client.index(indexRequestBuilder.build());
            System.out.println(indexResponse);
            
            // Make sure searching can provide the commits.
            RefreshResponse refreshResponse = client.indices().refresh();
            System.out.println(refreshResponse);
            
            // Attempt to search for commits with specific dateTime ranges.        
            SearchRequest.Builder searchRequestBuilder = new SearchRequest.Builder();
            searchRequestBuilder.q("dateTimeRange:[2022-07-27T12:12:12.121Z TO *]");
            SearchRequest searchRequest = searchRequestBuilder.build();
            SearchResponse<Commit> searchResponse = client.search(searchRequest, Commit.class);
            System.out.println(searchResponse);
            List<Hit<Commit>> hits = searchResponse.hits().hits();
            System.out.println("Hits");
            System.out.println("-----");
            for (Hit<Commit> hit : hits) {
                System.out.println(hit.source());
            }
            System.out.println("-----");
            assertEquals(2, hits.size());
    
        } finally {
            
            // Delete the first commit.
            DeleteRequest.Builder deleteRequestBuilder = new DeleteRequest.Builder();
            deleteRequestBuilder.index("commit");
            deleteRequestBuilder.id("commit20220727");
            DeleteRequest deleteRequest = deleteRequestBuilder.build();
            DeleteResponse deleteResponse = client.delete(deleteRequest);
            System.out.println(deleteResponse);
            
            // Delete the second commit.
            deleteRequestBuilder = new DeleteRequest.Builder();
            deleteRequestBuilder.index("commit");
            deleteRequestBuilder.id("commit20220728");
            deleteRequest = deleteRequestBuilder.build();
            deleteResponse = client.delete(deleteRequest);
            System.out.println(deleteResponse);
            
            // Delete index commit.
            DeleteIndexRequest.Builder deleteIndexRequestBuilder = new DeleteIndexRequest.Builder();
            deleteIndexRequestBuilder.index("commit");
            DeleteIndexRequest deleteIndexRequest = deleteIndexRequestBuilder.build();
            DeleteIndexResponse deleteIndexResponse = client.indices().delete(deleteIndexRequest);
            System.out.println(deleteIndexResponse);
        }
    }
}