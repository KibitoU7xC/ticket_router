import os
import json
from openai import OpenAI
from server.ticket_router_environment import TicketRouterEnvironment
from models import Action

def main():
    # Load HF token from environment variable (never hardcode secrets!)
    hf_token = os.environ.get("HF_TOKEN")
    
    client = OpenAI(
        base_url="https://router.huggingface.co/v1/", 
        api_key=hf_token
    )
    
    model_name = "Qwen/Qwen2.5-72B-Instruct" 

    env = TicketRouterEnvironment()
    obs = env.reset()
    
    print("--- Starting Ticket Router AI Test ---")
    
    max_steps = 10  # Safety limit to prevent infinite loops
    step = 0
    ticket_num = 1
    total_reward = 0.0
    
    while step < max_steps:
        step += 1
        print(f"\n[Ticket {ticket_num}] User says: '{obs.ticket_text}'")
        print(f"Routing Options: {obs.available_departments}")
        
        prompt = f"""
        You are a customer support routing agent.
        Ticket: "{obs.ticket_text}"
        Options: {obs.available_departments}
        Respond with ONLY a JSON object in this exact format: {{"department": "The chosen department"}}
        """
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            action = Action(department=result["department"])
            print(f"> Agent Decided To Route To: {action.department}")
            
            obs = env.step(action)
            reward = obs.reward
            done = obs.done
            total_reward += reward
            print(f"> Reward Given: {reward}")
            
            if reward > 0:
                print("> [CORRECT]")
                ticket_num += 1
            else:
                print("> [WRONG] Wrong department, retrying...")
            
            if done:
                if reward > 0:
                    print("\n*** All tickets resolved successfully! ***")
                else:
                    print("\n*** Too many failed attempts. Episode over. ***")
                break
                
        except Exception as e:
            print(f"API Error: Make sure your HF token is valid! Details: {e}")
            break
    
    print(f"\n--- Results ---")
    print(f"Total Reward: {total_reward}")
    print(f"Tickets Resolved: {env.state.total_resolved}/3")

if __name__ == "__main__":
    main()