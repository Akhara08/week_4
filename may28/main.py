import sympy as sp
import google.generativeai as genai

from autogen import AssistantAgent, UserProxyAgent, GroupChat, tools


# ====================
# Configure Gemini API
# ====================
genai.configure(api_key="AIzaSyDxsZ7llhXOT9HzG9bCPZ4p0xGxBEMNYAQ")  # Replace with your Gemini API key
model = genai.GenerativeModel("gemini-1.5-pro-latest")


# ====================
# Define the function for the SymPy tool
# ====================
def sympy_solver(problem: str):
    try:
        if "differentiate" in problem.lower():
            expr_str = problem.lower().replace("differentiate", "").strip()
            x = sp.symbols('x')
            expr = sp.sympify(expr_str)
            derivative = sp.diff(expr, x)
            return {
                "type": "derivative",
                "solution": derivative,
                "latex": sp.latex(derivative)
            }
        elif "=" in problem:
            lhs, rhs = problem.split("=")
            x = sp.symbols('x')
            eq = sp.Eq(sp.sympify(lhs.strip()), sp.sympify(rhs.strip()))
            sol = sp.solve(eq, x)
            return {
                "type": "equation",
                "solution": sol,
                "latex": f"x = {sp.latex(sol)}"
            }
        else:
            result = sp.sympify(problem).evalf()
            return {
                "type": "arithmetic",
                "solution": result,
                "latex": sp.latex(result)
            }
    except Exception as e:
        return {"error": str(e)}


# ====================
# Create SymPy tool by passing the function
# ====================
sympy_tool = tools.Tool(name="SymPyTool", func_or_tool=sympy_solver)


# ====================
# Agents
# ====================
class ProblemSolverAgent(AssistantAgent):
    def _get_tools(self):
        return [sympy_tool]

    async def on_message(self, message):
        problem = message.content
        solution = self._get_tools()[0](problem)

        if "error" in solution:
            response = f"Solver Error: {solution['error']}"
        else:
            response = f"Solution (LaTeX): {solution['latex']}"

        print(response)  # ✅ REPLACED send_message with print
        return solution



class VerifierAgent(AssistantAgent):
    def _init_(self, *args, **kwargs):
        super()._init_(*args, **kwargs)
        self.memory = {}  # add this to store state

    async def on_message(self, message):
        content = message.content

        lines = content.split("\n")
        problem_line = None
        latex_line = None
        for line in lines:
            if line.startswith("Problem:"):
                problem_line = line[len("Problem:"):].strip()
            if line.startswith("Solution (LaTeX):"):
                latex_line = line[len("Solution (LaTeX):"):].strip()

        if not problem_line:
            problem_line = self.memory.get("problem", "")
        if latex_line is None:
            latex_line = content

        prompt = f"""
You are a math verifier AI in a round-robin dialogue.

Given the math problem:

{problem_line}

The proposed solution (in LaTeX) is:

{latex_line}

Check if this solution is mathematically correct.
Reply with "Yes" or "No" and give reasoning.
If it's correct, end with "TERMINATE".
"""

        try:
            response = model.generate_content(prompt)
            reply = response.text.strip()
        except Exception as e:
            reply = f"Verifier API error: {e}"

        print(reply)  # ✅ Print instead of sending

        return reply


# ====================
# Main Round Robin Group Chat
# ====================
async def main():
    # Initialize agents
    solver = ProblemSolverAgent(name="SolverAgent")
    verifier = VerifierAgent(name="VerifierAgent")
    
    problem = "Find the derivative of sin(x)."
    verifier.memory["problem"] = problem



    # Create group chat with round robin order: solver → verifier → solver → ...
    group_chat = GroupChat(agents=[solver, verifier])

    print("Welcome to the Smart Math Tutor! Type your math problem, or 'exit' to quit.")

    while True:
        problem = input("\nEnter math problem > ").strip()
        if problem.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Store problem for verifier to access if needed
        verifier.memory["problem"] = problem

        iteration = 1
        while True:
            print(f"\n--- Round {iteration} ---")

            # Solver turn
            solver_response = await solver.on_message(type("Msg", (), {"content": problem}))
            if isinstance(solver_response, dict) and "error" in solver_response:
                print("Solver Error:", solver_response["error"])
                break
            print(f"Solver proposed solution: {solver_response.get('latex', '')}")

            # Verifier turn: give problem and solution latex
            verifier_input = f"Problem: {problem}\nSolution (LaTeX): {solver_response.get('latex', '')}"
            verifier_response = await verifier.on_message(type("Msg", (), {"content": verifier_input}))

            print(f"Verifier response: {verifier_response}")

            if "TERMINATE" in verifier_response:
                print("✅ Solution verified! Ending session.")
                break
            else:
                print("⚠ Verifier found issues. Re-solving...")
                iteration += 1


# ====================
# Run the async main
# ====================
import asyncio

if __name__ == "_main_":
    asyncio.run(main())