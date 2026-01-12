use std::rc::Rc;

use yew::prelude::*;
use yew::{html, Component, Context, Html};
use gloo_timers::future::sleep;
use std::time::Duration;

use crate::ai::AI;
use crate::logic::{Board, Direction, Pawn, Position, Color};

pub enum Msg {
    PawnClick(usize),
    Left,
    Right,
    Up,
    Down,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
    Restart,
    AiGreen,
    AiYellow,
    AiMoveReady(Option<(usize, Direction)>),
}

pub struct App {
    board: Board,
    state: Rc<AppState>,
    direction: Direction,
    ai: Option<Color>,
    ai_depth: usize,
    ai_thinking: bool,
}

#[derive(Clone, PartialEq)]
pub struct AppState {
    pawn_clicked: Callback<usize>,
    last_clicked: Option<usize>
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let pawn_clicked = ctx.link().callback(Msg::PawnClick);
        let state = Rc::new(AppState {
            pawn_clicked,
            last_clicked: None
        });

        let board = Board::default_new();
        Self {
            board,
            state,
            direction: Direction::Up,
            ai: None,
            ai_depth: 5,
            ai_thinking: false,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::PawnClick(pawn_index) => {
                let shared_state = Rc::make_mut(&mut self.state);
                shared_state.last_clicked = Some(pawn_index);

                let mut new_board = self.board.clone();
                if new_board.move_pawn_until_blocked(pawn_index, &self.direction) {
                    self.board = new_board;
                    if let Some(ai_color) = &self.ai {
                        // Set AI thinking state
                        self.ai_thinking = true;
                        
                        // Clone data needed for the async task
                        let board = self.board.clone();
                        let ai_color = ai_color.clone();
                        let ai_depth = self.ai_depth;
                        
                        // Spawn async task to calculate AI move
                        let link = ctx.link().clone();
                        wasm_bindgen_futures::spawn_local(async move {
                            // Small delay to allow browser to render player's move first
                            sleep(Duration::from_millis(50)).await;
                            let ai_move = AI::new(ai_color, board, ai_depth).ai_play();
                            link.send_message(Msg::AiMoveReady(ai_move));
                        });
                        
                        return true;
                    }
                }
            }
            Msg::AiMoveReady(ai_move) => {
                self.ai_thinking = false;
                if let Some((ai_pawn_index, ai_direction)) = ai_move {
                    self.board.move_pawn_until_blocked(ai_pawn_index, &ai_direction);
                }
            }
            Msg::Up => {
                self.direction = Direction::Up;
            }
            Msg::Down => {
                self.direction = Direction::Down;
            }
            Msg::Left => {
                self.direction = Direction::Left;
            }
            Msg::Right => {
                self.direction = Direction::Right;
            }
            Msg::UpLeft => {
                self.direction = Direction::UpLeft;
            }
            Msg::UpRight => {
                self.direction = Direction::UpRight;
            }
            Msg::DownLeft => {
                self.direction = Direction::DownLeft;
            }
            Msg::DownRight => {
                self.direction = Direction::DownRight;
            }
            Msg::Restart => {
                self.board = Board::default_new();
                self.ai = None;
                self.ai_thinking = false;
            }
            Msg::AiGreen => {
                self.ai = Some(Color::Yellow);
                
                // Set AI thinking state
                self.ai_thinking = true;
                
                // Clone data needed for the async task
                let board = self.board.clone();
                let ai_depth = self.ai_depth;
                
                // Spawn async task to calculate AI move
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    // Small delay to allow browser to render board state first
                    sleep(Duration::from_millis(50)).await;
                    let ai_move = AI::new(Color::Yellow, board, ai_depth).ai_play();
                    link.send_message(Msg::AiMoveReady(ai_move));
                });
                
                return true;
            }
            Msg::AiYellow => {
                self.ai = Some(Color::Green);
                
                // Set AI thinking state
                self.ai_thinking = true;
                
                // Clone data needed for the async task
                let board = self.board.clone();
                let ai_depth = self.ai_depth;
                
                // Spawn async task to calculate AI move
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    // Small delay to allow browser to render board state first
                    sleep(Duration::from_millis(50)).await;
                    let ai_move = AI::new(Color::Green, board, ai_depth).ai_play();
                    link.send_message(Msg::AiMoveReady(ai_move));
                });
                
                return true;
            }
        }
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let app_state = self.state.clone();
        let next_player_text = match self.board.winner() {
            Some(Color::Green) => "Green wins!".to_string(),
            Some(Color::Yellow) => "Yellow wins!".to_string(),
            None => {
                if self.ai_thinking {
                    "🤔 AI is thinking...".to_string()
                } else {
                    match self.board.next_player {
                        Some(Color::Green) => "Green's turn".to_string(),
                        Some(Color::Yellow) => "Yellow's turn".to_string(),
                        None => String::new(),
                    }
                }
            }
        };
        
        let is_active = |dir: &Direction| -> &str {
            if *dir == self.direction { "background-color: #004d2f;" } else { "" }
        };
        
        let button_disabled = if self.ai_thinking { "disabled" } else { "" };
        let button_opacity = if self.ai_thinking { "opacity: 0.5; cursor: not-allowed;" } else { "" };
        
        html! {
            <ContextProvider<Rc<AppState>> context={app_state}>
                <div style="display: flex; flex-direction: column; align-items: left; gap: 10px; margin-bottom: 20px;">
                    <div style="display: flex; gap: 10px; justify-content: left;">
                        <button {button_disabled} style={format!("width: 50px; height: 50px; font-size: 20px; {}; {}", is_active(&Direction::UpLeft), button_opacity)} onclick={ctx.link().callback(|_| Msg::UpLeft)}>{ "↖" }</button>
                        <button {button_disabled} style={format!("width: 50px; height: 50px; font-size: 20px; {}; {}", is_active(&Direction::Up), button_opacity)} onclick={ctx.link().callback(|_| Msg::Up)}>{ "↑" }</button>
                        <button {button_disabled} style={format!("width: 50px; height: 50px; font-size: 20px; {}; {}", is_active(&Direction::UpRight), button_opacity)} onclick={ctx.link().callback(|_| Msg::UpRight)}>{ "↗" }</button>
                    </div>
                    <div style="display: flex; gap: 10px; justify-content: left;">
                        <button {button_disabled} style={format!("width: 50px; height: 50px; font-size: 20px; {}; {}", is_active(&Direction::Left), button_opacity)} onclick={ctx.link().callback(|_| Msg::Left)}>{ "←" }</button>
                        <div style="width: 50px; height: 50px;"></div>
                        <button {button_disabled} style={format!("width: 50px; height: 50px; font-size: 20px; {}; {}", is_active(&Direction::Right), button_opacity)} onclick={ctx.link().callback(|_| Msg::Right)}>{ "→" }</button>
                    </div>
                    <div style="display: flex; gap: 10px; justify-content: left;">
                        <button {button_disabled} style={format!("width: 50px; height: 50px; font-size: 20px; {}; {}", is_active(&Direction::DownLeft), button_opacity)} onclick={ctx.link().callback(|_| Msg::DownLeft)}>{ "↙" }</button>
                        <button {button_disabled} style={format!("width: 50px; height: 50px; font-size: 20px; {}; {}", is_active(&Direction::Down), button_opacity)} onclick={ctx.link().callback(|_| Msg::Down)}>{ "↓" }</button>
                        <button {button_disabled} style={format!("width: 50px; height: 50px; font-size: 20px; {}; {}", is_active(&Direction::DownRight), button_opacity)} onclick={ctx.link().callback(|_| Msg::DownRight)}>{ "↘" }</button>
                    </div>
                </div>
                <h2>{ next_player_text }</h2>
                <div>
                    <button {button_disabled} onclick={ctx.link().callback(|_| Msg::Restart)}>{ "Restart Game" }</button>
                    <button {button_disabled} onclick={ctx.link().callback(|_| Msg::AiGreen)}>{ "Play against AI as Green" }</button>
                    <button {button_disabled} onclick={ctx.link().callback(|_| Msg::AiYellow)}>{ "Play against AI as Yellow" }</button>
                </div>
                <BoardView board={self.board.clone()} />
            </ContextProvider<Rc<AppState>>>
        }
    }

}

pub struct PawnView{
    state: Rc<AppState>,
    _listener: ContextHandle<Rc<AppState>>
}

#[derive(Clone, Properties, PartialEq)]
pub struct PawnComponent {
    pub pawn: Pawn,
    pub position: Position,
    pub index: usize
}

const SCALING: u32 = 80;

const MARGIN: u32 = 5;

pub enum PawnMsg {
    ContextChanged(Rc<AppState>),
}

impl Component for PawnView {
    type Message = PawnMsg;
    type Properties = PawnComponent;
 
    fn create(ctx: &Context<Self>) -> Self {
        let (state, _listener) = ctx
            .link()
            .context::<Rc<AppState>>(ctx.link().callback(PawnMsg::ContextChanged))
            .expect("context to be set");

        Self { state, _listener }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            PawnMsg::ContextChanged(state) => {
                self.state = state;
                true
            }
        }
    }
 
    fn view(&self, ctx: &Context<Self>) -> Html {
        let my_index = ctx.props().index;
        let onclick = self.state.pawn_clicked.reform(move |_| my_index);
        html! {
            <div
                onclick={onclick}
                style={format!(
                    "width: {}px; height: {}px; background-color: {}; position: absolute; top: {}px; left: {}px; border-radius: 50%; border: 2px solid black; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; color: black;",
                    SCALING - MARGIN * 2,
                    SCALING - MARGIN * 2,
                    match ctx.props().pawn.color {
                        Color::Green => "green",
                        Color::Yellow => "yellow",
                    },
                    u32::from(ctx.props().position.row) * SCALING + MARGIN - 1,
                    u32::from(ctx.props().position.column) * SCALING + MARGIN - 1,
                )}
            >
                //{ctx.props().index}
            </div>
        }
    }
}

pub struct BoardView;

#[derive(Clone, Properties, PartialEq)]
pub struct BoardComponent {
    pub board: Board,
}
 
impl Component for BoardView {
    type Message = ();
    type Properties = BoardComponent;
 
    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }
 
    fn view(&self, ctx: &Context<Self>) -> Html {
        let mut pawns = Vec::new();
        for (index, pawn) in ctx.props().board.pawns.iter().enumerate() {
            pawns.push(html! {
                <PawnView pawn={pawn.clone()} position={pawn.position.clone()} index={index} />
            });
        }
        html! {
            <div style={format!(
                "position: relative; top: {}px; width: {}px; height: {}px; background-image: linear-gradient(0deg, #e0e0e0 1px, transparent 1px), linear-gradient(90deg, #e0e0e0 1px, transparent 1px); background-size: {}px {}px; background-position: 0 0; border-top: 1px solid #e0e0e0; border-left: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0; border-bottom: 1px solid #e0e0e0;",
                MARGIN,
                SCALING * ctx.props().board.number_of_columns as u32,
                SCALING * ctx.props().board.number_of_rows as u32,
                SCALING, SCALING
            )}>
                {pawns}
            </div>
        }
    }
}
