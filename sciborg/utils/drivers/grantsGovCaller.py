import requests
from typing import Dict, Union, Optional

class GrantsGovAPIError(Exception):
    """Custom exception for Grants.gov API related errors."""
    pass

def search_opportunities(
    rows: int = 10,
    keyword: str = "20231011",
    oppNum: str = "TEST-PTS-20231011-OPP1",
    eligibilities: str = "",
    agencies: str = "",
    oppStatuses: str = "forecasted|posted",
    aln: str = "",
    fundingCategories: str = ""
) -> Dict:
    """
    Search for grant opportunities using the Grants.gov Search2 endpoint.
    
    Args:
        rows: Number of results to return (default: 10)
        keyword: Keyword used in the search (default: "20231011")
        oppNum: Opportunity number filter (default: "TEST-PTS-20231011-OPP1")
        eligibilities: Comma-separated list of eligibility codes
        agencies: Comma-separated list of agency codes
        oppStatuses: Opportunity statuses, separated by "|" (e.g., "forecasted|posted")
        aln: Assistance Listing Number filter
        fundingCategories: Comma-separated list of funding category codes
        
    Returns:
        Dict containing the parsed JSON response from the endpoint
        
    Raises:
        GrantsGovAPIError: If there's an error communicating with the Grants.gov API
        ValueError: If invalid parameters are provided
    """
    url = "https://api.grants.gov/v1/api/search2"
    
    # Input validation
    if rows < 1 or rows > 1000:  # Reasonable upper limit
        raise ValueError("Number of rows must be between 1 and 1000")
    if not isinstance(keyword, str):
        raise ValueError("Keyword must be a string")
    if not isinstance(oppNum, str):
        raise ValueError("Opportunity number must be a string")
    
    payload = {
        "rows": rows,
        "keyword": keyword,
        "oppNum": oppNum,
        "eligibilities": eligibilities,
        "agencies": agencies,
        "oppStatuses": oppStatuses,
        "aln": aln,
        "fundingCategories": fundingCategories
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        raise GrantsGovAPIError(f"Request timeout: The API did not respond within 30 seconds")
    except requests.HTTPError as http_err:
        raise GrantsGovAPIError(f"HTTP error occurred: {http_err}")
    except requests.RequestException as req_err:
        raise GrantsGovAPIError(f"Error making request: {req_err}")
    except ValueError as val_err:
        raise GrantsGovAPIError(f"Error parsing JSON response: {val_err}")

def fetch_opportunity(
    opportunity_id: Union[int, str],
    base_url: str = "https://api.grants.gov/v1/api/fetchOpportunity"
) -> Dict:
    """
    Fetch detailed information for a specific opportunity from Grants.gov.
    
    Args:
        opportunity_id: The unique ID of the opportunity to fetch
        base_url: The endpoint URL for fetching opportunity details
        
    Returns:
        Dict containing the opportunity details
        
    Raises:
        GrantsGovAPIError: If there's an error communicating with the Grants.gov API
        ValueError: If opportunity_id is invalid
    """
    if not opportunity_id:
        raise ValueError("opportunity_id cannot be empty")

    payload = {"opportunityId": opportunity_id}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(base_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        raise GrantsGovAPIError(f"Request timeout: The API did not respond within 30 seconds")
    except requests.HTTPError as http_err:
        raise GrantsGovAPIError(f"HTTP error occurred: {http_err}")
    except requests.RequestException as req_err:
        raise GrantsGovAPIError(f"Error making request: {req_err}")
    except ValueError as val_err:
        raise GrantsGovAPIError(f"Error parsing JSON response: {val_err}")

if __name__ == "__main__":
    # Example usage with error handling
    try:
        # Search for opportunities
        search_results = search_opportunities(rows=5)
        print("Search2 Endpoint Response:")
        print(search_results)
        
        # Fetch specific opportunity
        opp_id = 289999
        opportunity_details = fetch_opportunity(opp_id)
        print("\nFetch Opportunity Response:")
        print(opportunity_details)
        
    except GrantsGovAPIError as api_err:
        print(f"Grants.gov API error: {api_err}")
    except ValueError as val_err:
        print(f"Invalid input: {val_err}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")


#TODO: next steps
# https://www.grants.gov/api/api-guide